"""Fit every extension model before evaluating any selected model on test data."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import Dataset

from .extension_config import (
    FAMILIES,
    ExtensionConfig,
    Task,
    make_tasks,
    parse_arguments,
)
from .extension_data import make_loader, make_probe_samples, make_splits, task_targets
from .extension_features import extract_features
from .extension_probes import classification_metrics, fit_probe
from .extension_reporting import (
    paired_differences,
    plot_confusions,
    plot_probe_curves,
    save_csv,
    save_json,
    summarize,
)
from .extension_training import (
    choose_device,
    cpu_state,
    evaluate_cnn,
    freeze,
    make_models,
    train_cnn,
)
from .models import SmallCNN


def record_environment() -> dict:
    """Record software versions and hashes of extension and shared model code."""
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "threadpoolctl",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    directory = Path(__file__).parent
    paths = sorted(directory.glob("extension_*.py")) + [directory / "models.py"]
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "source_sha256": hashes,
    }


def fit_seed(
    seed: int,
    dataset: Dataset,
    original_labels: np.ndarray,
    splits: dict,
    tasks: list[Task],
    config: ExtensionConfig,
    device: torch.device,
) -> list[dict]:
    """Train both supervision conditions and fit all paired probes for one seed.

    Args:
        seed: Initialization/probe-sampling seed.
        dataset: Official training dataset; no test data are accepted here.
        original_labels: Original labels indexed by dataset position.
        splits: Fixed training and validation indices.
        tasks: Downstream probe definitions.
        config: Full configuration.
        device: CNN training/extraction device.

    Returns:
        Saved probe metadata; each payload also contains its fitted scaler.
    """
    directory = config.output_dir / f"seed_{seed}"
    probe_dir = directory / "probes"
    probe_dir.mkdir(parents=True)
    pool_ids, samples = make_probe_samples(
        splits["cnn_train"], original_labels, tasks, config.probe_sizes, seed
    )
    arrays = {
        f"{task_name}__n{budget}": pool_ids[positions]
        for (task_name, budget), positions in samples.items()
    }
    np.savez_compressed(
        directory / "probe_samples.npz", feature_pool=pool_ids, **arrays
    )
    models = make_models(seed)
    freeze(models["random"])
    torch.save(cpu_state(models["random"]), directory / "random_cnn.pt")
    for family in ("fine", "coarse"):
        train_cnn(
            models[family], family, dataset, splits, config, device, seed, directory
        )

    manifest = []
    train_loader = make_loader(dataset, pool_ids, config, device)
    val_loader = make_loader(dataset, splits["probe_validation"], config, device)
    for family in FAMILIES:
        print(f"Seed {seed}: fitting {family} probes...")
        model = models.get(family)
        train_features, train_original = extract_features(model, train_loader, device)
        val_features, val_original = extract_features(model, val_loader, device)
        if not np.array_equal(train_original, original_labels[pool_ids]):
            raise RuntimeError("Feature rows do not match the saved sampling plan.")
        for task in tasks:
            val_mask, val_targets = task_targets(val_original, task)
            for budget in config.probe_sizes:
                positions = samples[(task.name, budget)]
                mask, targets = task_targets(train_original[positions], task)
                if not mask.all():
                    raise RuntimeError(
                        "The sampling plan contains an unrelated task class."
                    )
                counts = np.bincount(targets, minlength=len(task.groups))
                if not np.all(counts == budget):
                    raise RuntimeError("Probe training samples are not balanced.")
                for layer, feature_matrix in train_features.items():
                    selected = fit_probe(
                        feature_matrix[positions],
                        targets,
                        val_features[layer][val_mask],
                        val_targets,
                        config,
                        seed,
                        len(task.groups),
                    )
                    name = f"{task.name}__{family}_{layer}__n{budget}.joblib"
                    record = {
                        "seed": seed,
                        "task": task.name,
                        "task_category": task.category,
                        "family": family,
                        "layer": layer,
                        "examples_per_class": budget,
                        "n_probe_train": len(targets),
                        "n_probe_validation": len(val_targets),
                        "n_cnn_train": len(splits["cnn_train"])
                        if family in ("fine", "coarse")
                        else 0,
                        "feature_count": feature_matrix.shape[1],
                        "selected_c": selected["selected_c"],
                        "validation_accuracy": selected["validation"]["accuracy"],
                        "validation_macro_f1": selected["validation"]["macro_f1"],
                        "validation_class_counts": np.bincount(
                            val_targets, minlength=len(task.groups)
                        ).tolist(),
                        "training_majority_class": int(counts.argmax()),
                        "probe_file": name,
                        "class_names": task.class_names,
                        "original_label_groups": task.groups,
                    }
                    payload = {
                        **selected,
                        "metadata": record,
                        "training_indices": pool_ids[positions],
                    }
                    joblib.dump(payload, probe_dir / name)
                    manifest.append(record)
        del train_features, val_features
        gc.collect()
    save_json(directory / "probe_manifest.json", manifest)
    return manifest


def load_model(family: str, checkpoint: Path) -> SmallCNN:
    """Load a tensor-only extension checkpoint into the appropriate head shape."""
    model = SmallCNN()
    if family == "coarse":
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    freeze(model)
    return model


def evaluate_seed(
    seed: int,
    manifest: list[dict],
    dataset: Dataset,
    splits: dict,
    tasks: list[Task],
    config: ExtensionConfig,
    device: torch.device,
) -> tuple[list, list]:
    """Evaluate saved selected probes and CNNs, without performing any fitting.

    Args:
        seed: Seed identifying saved artifacts.
        manifest: Preselected probes from the fitting phase.
        dataset: Official test dataset, first used for metrics in this phase.
        splits: Test indices selected before training.
        tasks: Fixed task definitions.
        config: Configuration and artifact root.
        device: CNN inference device.

    Returns:
        Probe records and CNN records. Each probe stores task-specific test
        class counts, including natural imbalance in pooled contrasts.
    """
    directory = config.output_dir / f"seed_{seed}"
    loader = make_loader(dataset, splits["test"], config, device)
    task_lookup = {task.name: task for task in tasks}
    rows, cnn_rows = [], []
    baselines_done = set()
    for family in FAMILIES:
        model = (
            None
            if family == "pixels"
            else load_model(family, directory / f"{family}_cnn.pt")
        )
        if family in ("fine", "coarse"):
            scores = evaluate_cnn(
                model.to(device), loader, device, family, config.coarse_target
            )
            cnn_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "n_test": len(splits["test"]),
                    **scores,
                }
            )
        features, original_labels = extract_features(model, loader, device)
        for record in manifest:
            if record["family"] != family:
                continue
            task = task_lookup[record["task"]]
            mask, targets = task_targets(original_labels, task)
            selected = joblib.load(directory / "probes" / record["probe_file"])
            inputs = selected["scaler"].transform(features[record["layer"]][mask])
            scores = classification_metrics(
                targets, selected["classifier"].predict(inputs), len(task.groups)
            )
            row = {
                key: value
                for key, value in record.items()
                if key not in ("probe_file", "training_majority_class")
            }
            row.update(
                {
                    "n_test": len(targets),
                    "test_class_counts": np.bincount(
                        targets, minlength=len(task.groups)
                    ).tolist(),
                    **{f"test_{key}": value for key, value in scores.items()},
                }
            )
            rows.append(row)
            key = (task.name, record["examples_per_class"])
            if key not in baselines_done:
                scores = classification_metrics(
                    targets,
                    np.full_like(targets, record["training_majority_class"]),
                    len(task.groups),
                )
                baseline = {
                    **row,
                    "family": "constant",
                    "layer": "constant",
                    "feature_count": 0,
                    "n_cnn_train": 0,
                    "selected_c": None,
                    "validation_accuracy": None,
                    "validation_macro_f1": None,
                    **{f"test_{name}": value for name, value in scores.items()},
                }
                rows.append(baseline)
                baselines_done.add(key)
            del inputs
        del features, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(directory / "test_results.json", rows)
    save_csv(directory / "test_results.csv", rows)
    plot_confusions(rows, tasks, directory)
    return rows, cnn_rows


def run(
    config: ExtensionConfig,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> None:
    """Run the extension, accepting injected datasets only for automated tests.

    Args:
        config: Validated configuration.
        train_dataset: Optional test fixture with a targets attribute.
        test_dataset: Matching optional test fixture.

    Raises:
        FileExistsError: If the output directory is not empty.
        ValueError: If only one dataset fixture is provided.
        RuntimeError: If any fitting or experimental control fails.
    """
    if (train_dataset is None) != (test_dataset is None):
        raise ValueError("Provide both test fixtures or neither.")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    torch.set_num_threads(config.threads)
    start = time.perf_counter()
    save_json(config.output_dir / "config.json", config)
    save_json(config.output_dir / "environment.json", record_environment())
    save_json(
        config.output_dir / "status.json", {"stage": "preparing", "device": str(device)}
    )
    try:
        if train_dataset is None:
            from torchvision import datasets, transforms

            transform = transforms.ToTensor()
            train_dataset = datasets.FashionMNIST(
                config.data_dir, train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                config.data_dir, train=False, download=True, transform=transform
            )
        labels = np.asarray(train_dataset.targets, dtype=np.int64)
        test_labels = np.asarray(test_dataset.targets, dtype=np.int64)
        splits = make_splits(labels, test_labels, config)
        np.savez_compressed(config.output_dir / "split_indices.npz", **splits)
        tasks = make_tasks(config.include_cross_pairs, config.coarse_target)
        save_json(config.output_dir / "tasks.json", [asdict(task) for task in tasks])
        save_json(
            config.output_dir / "status.json",
            {"stage": "fitting", "device": str(device)},
        )
        print(
            f"Using {device}; {len(tasks)} tasks; seeds {config.seeds}; budgets {config.probe_sizes}"
        )
        manifests = {}
        with threadpool_limits(limits=config.threads):
            for seed in config.seeds:
                manifests[seed] = fit_seed(
                    seed, train_dataset, labels, splits, tasks, config, device
                )
            # Every seed's CNN and probe choices are fixed before any test metrics.
            save_json(
                config.output_dir / "status.json",
                {"stage": "evaluating", "device": str(device)},
            )
            all_rows, cnn_rows = [], []
            for seed in config.seeds:
                print(f"Evaluating frozen seed {seed} on test data...")
                rows, cnn = evaluate_seed(
                    seed, manifests[seed], test_dataset, splits, tasks, config, device
                )
                all_rows.extend(rows)
                cnn_rows.extend(cnn)
        summary = summarize(all_rows)
        paired = paired_differences(all_rows)
        save_csv(config.output_dir / "probe_results.csv", all_rows)
        save_json(config.output_dir / "probe_results.json", all_rows)
        save_csv(config.output_dir / "summary.csv", summary)
        save_csv(config.output_dir / "paired_differences.csv", paired)
        save_csv(config.output_dir / "paired_summary.csv", summarize(paired))
        save_csv(config.output_dir / "cnn_test_metrics.csv", cnn_rows)
        figures = config.output_dir / "figures"
        figures.mkdir()
        for task in tasks:
            for metric in ("macro_f1", "balanced_accuracy"):
                plot_probe_curves(summary, task, figures, metric)
        elapsed = time.perf_counter() - start
        save_json(
            config.output_dir / "status.json",
            {
                "stage": "complete",
                "seconds": elapsed,
                "device": str(device),
                "dataset_class": type(train_dataset).__name__,
                "probe_rows": len(all_rows),
            },
        )
        print(
            f"Finished in {elapsed:.1f} seconds. Results: {config.output_dir.resolve()}"
        )
    except Exception as exc:
        save_json(
            config.output_dir / "status.json",
            {"stage": "failed", "error": str(exc), "device": str(device)},
        )
        raise


def main() -> None:
    """Run the extension from the command line, including Windows-safe workers."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(parse_arguments())
