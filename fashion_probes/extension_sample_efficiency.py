"""Low-data probes for the add-on using frozen extension checkpoints.

The experiment reuses one completed extension run, draws nested probe-training
samples with exact total sizes, projects every representation to a common
dimension, and fixes every probe before the official test split is loaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter
from threadpoolctl import threadpool_limits
from torch.utils.data import Dataset

from .extension_config import FAMILIES, LAYERS, Task, make_tasks
from .extension_data import make_loader, task_targets
from .extension_features import extract_features
from .extension_pipeline import load_model
from .extension_probes import classification_metrics, fit_probe
from .extension_reporting import save_csv, save_json
from .extension_training import choose_device
from .reporting import (
    BASELINE_COLOR,
    GRID,
    INK,
    MUTED_INK,
    PAPER,
    PIXEL_COLOR,
    RANDOM_COLOR,
    TRAINED_COLOR,
    _save_figure,
    _style_axis,
)
from .sample_efficiency import random_project


DEFAULT_SAMPLE_SIZES = (10, 20, 30, 50, 80, 110, 150, 200)
DEFAULT_REPEATS = 10
DEFAULT_PROJECTION_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_CONTROLLED_DIM = 32
COARSE_COLOR = "#B07A2A"  # muted ochre, distinct from the shared palette
CONDITION_ORDER = (
    "constant",
    "pixels",
    "random_conv1",
    "fine_conv1",
    "coarse_conv1",
    "random_conv2",
    "fine_conv2",
    "coarse_conv2",
    "random_fc",
    "fine_fc",
    "coarse_fc",
)


@dataclass(frozen=True)
class ExtensionProbeConfig:
    """Settings for the frozen-checkpoint add-on probe experiment."""

    source_dir: Path
    output_dir: Path
    data_dir: Path | None
    source_seed: int
    seed: int
    sample_sizes: tuple[int, ...]
    repeats: int
    controlled_dim: int
    projection_seeds: tuple[int, ...]
    feature_batch_size: int
    num_workers: int
    requested_device: str
    threads: int


@dataclass
class FittedExtensionProbe:
    """One selected linear probe and its complete experimental identity."""

    repeat: int
    sampling_seed: int
    task: Task
    sample_size: int
    projection_seed: int
    family: str
    layer: str
    original_feature_count: int
    training_indices: np.ndarray
    training_class_counts: np.ndarray
    selected: dict[str, Any]

    @property
    def condition(self) -> str:
        """Return the stable display/output identifier for this probe."""
        return "pixels" if self.family == "pixels" else f"{self.family}_{self.layer}"


def parse_arguments(argv: Sequence[str] | None = None) -> ExtensionProbeConfig:
    """Parse command-line settings for the prespecified low-data design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("results/extension_full")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extension_sample_efficiency"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Fashion-MNIST root; defaults to the source run's data_dir.",
    )
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--controlled-dim", type=int, default=DEFAULT_CONTROLLED_DIM)
    parser.add_argument(
        "--projection-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_PROJECTION_SEEDS,
    )
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        dest="requested_device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--threads", type=int, default=2)
    values = vars(parser.parse_args(argv))
    values["sample_sizes"] = tuple(values["sample_sizes"])
    values["projection_seeds"] = tuple(values["projection_seeds"])
    config = ExtensionProbeConfig(**values)
    validate_config(config)
    return config


def validate_config(config: ExtensionProbeConfig) -> None:
    """Reject ambiguous or infeasible low-data configurations."""
    if not config.sample_sizes:
        raise ValueError("At least one sample size is required.")
    if tuple(sorted(set(config.sample_sizes))) != config.sample_sizes:
        raise ValueError("sample_sizes must be strictly increasing and unique.")
    if min(config.sample_sizes) < 2:
        raise ValueError("Every task needs at least two probe-training examples.")
    if config.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if config.controlled_dim <= 0:
        raise ValueError("controlled_dim must be positive.")
    if not config.projection_seeds:
        raise ValueError("At least one projection seed is required.")
    if len(set(config.projection_seeds)) != len(config.projection_seeds):
        raise ValueError("projection_seeds must be unique.")
    if config.feature_batch_size <= 0 or config.threads <= 0:
        raise ValueError("feature_batch_size and threads must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    try:
        if config.source_dir.resolve() == config.output_dir.resolve():
            raise ValueError("output_dir must differ from source_dir.")
    except OSError:
        pass


def nested_task_samples(
    train_ids: np.ndarray,
    labels: np.ndarray,
    tasks: Sequence[Task],
    sample_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, dict[tuple[int, str, int], np.ndarray]]:
    """Draw nested task samples with exact totals and near-equal target counts.

    The target classes are interleaved, so every prefix differs by at most one
    example across target classes. Multi-original-class groups are interleaved
    internally by the same rule. Returned selections index the sorted union
    feature pool and are shared by every representation and projection.
    """
    available = np.asarray(train_ids)
    original = np.asarray(labels)
    sizes = tuple(sample_sizes)
    if available.ndim != 1 or not np.issubdtype(available.dtype, np.integer):
        raise ValueError("train_ids must be a one-dimensional integer array.")
    if len(np.unique(available)) != len(available):
        raise ValueError("train_ids must not contain duplicates.")
    if np.any(available < 0) or np.any(available >= len(original)):
        raise ValueError("train_ids contain an out-of-range dataset index.")
    if not sizes or tuple(sorted(set(sizes))) != sizes or min(sizes) < 2:
        raise ValueError("sample_sizes must be increasing, unique, and at least two.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    selections: dict[tuple[int, str, int], np.ndarray] = {}
    largest = sizes[-1]
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        original_pools = {
            label: rng.permutation(available[original[available] == label])
            for label in range(10)
        }
        group_pools: dict[tuple[int, ...], np.ndarray] = {}
        for task in tasks:
            for group in task.groups:
                if group in group_pools:
                    continue
                group_order = rng.permutation(group)
                depth = min(len(original_pools[int(label)]) for label in group_order)
                group_pools[group] = np.stack(
                    [original_pools[int(label)][:depth] for label in group_order],
                    axis=1,
                ).reshape(-1)

        for task in tasks:
            target_order = rng.permutation(len(task.groups))
            target_depth = min(len(group_pools[group]) for group in task.groups)
            ordered = np.stack(
                [group_pools[task.groups[int(target)]][:target_depth] for target in target_order],
                axis=1,
            ).reshape(-1)
            if len(ordered) < largest:
                raise ValueError(
                    f"Task {task.name!r} has {len(ordered)} eligible examples; "
                    f"{largest} are required."
                )
            for size in sizes:
                selections[(repeat, task.name, size)] = ordered[:size].copy()

    pool_ids = np.unique(np.concatenate(list(selections.values())))
    positions = {
        key: np.searchsorted(pool_ids, ids) for key, ids in selections.items()
    }
    return pool_ids, positions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(config: ExtensionProbeConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and validate the source configuration, split, and checkpoints."""
    seed_dir = config.source_dir / f"seed_{config.source_seed}"
    required = (
        config.source_dir / "config.json",
        config.source_dir / "split_indices.npz",
        seed_dir / "random_cnn.pt",
        seed_dir / "fine_cnn.pt",
        seed_dir / "coarse_cnn.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifacts: " + ", ".join(missing))
    with (config.source_dir / "config.json").open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    if config.source_seed not in source_config.get("seeds", []):
        raise ValueError("source_seed was not part of the completed source run.")
    for field in ("probe_cs", "probe_max_iter"):
        if field not in source_config:
            raise ValueError(f"Source config is missing {field!r}.")
    with np.load(config.source_dir / "split_indices.npz") as archive:
        required_splits = {"cnn_train", "probe_validation", "test"}
        if not required_splits <= set(archive.files):
            raise ValueError("Source split archive is incomplete.")
        splits = {name: np.asarray(archive[name]).copy() for name in archive.files}
    for name in required_splits:
        indices = splits[name]
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Source split {name!r} must be a 1D integer array.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Source split {name!r} contains duplicates.")
    return source_config, splits


def _loader_config(config: ExtensionProbeConfig) -> SimpleNamespace:
    """Return only the loader fields used by ``extension_data.make_loader``."""
    return SimpleNamespace(
        batch_size=config.feature_batch_size,
        feature_batch_size=config.feature_batch_size,
        num_workers=config.num_workers,
    )


def _loader(
    dataset: Dataset,
    ids: np.ndarray,
    config: ExtensionProbeConfig,
    device: torch.device,
    seed: int,
):
    return make_loader(
        dataset,
        ids,
        _loader_config(config),
        device,
        shuffle=False,
        seed=seed,
    )


def _model_for_family(
    family: str, config: ExtensionProbeConfig
) -> torch.nn.Module | None:
    """Load one frozen add-on checkpoint, or return None for raw pixels."""
    if family == "pixels":
        return None
    checkpoint = (
        config.source_dir
        / f"seed_{config.source_seed}"
        / f"{family}_cnn.pt"
    )
    return load_model(family, checkpoint)


def fit_all_probes(
    train_dataset: Dataset,
    train_labels: np.ndarray,
    splits: dict[str, np.ndarray],
    tasks: list[Task],
    source_config: dict[str, Any],
    config: ExtensionProbeConfig,
    device: torch.device,
) -> tuple[
    list[FittedExtensionProbe],
    dict[tuple[int, str, int], np.ndarray],
    np.ndarray,
]:
    """Fit and select all probes without accepting test data as an argument."""
    pool_ids, samples = nested_task_samples(
        splits["cnn_train"],
        train_labels,
        tasks,
        config.sample_sizes,
        config.repeats,
        config.seed,
    )
    train_loader = _loader(
        train_dataset, pool_ids, config, device, config.seed + 1000
    )
    val_loader = _loader(
        train_dataset,
        splits["probe_validation"],
        config,
        device,
        config.seed + 1001,
    )
    expected_train_labels = train_labels[pool_ids]
    expected_val_labels = train_labels[splits["probe_validation"]]
    fit_config = SimpleNamespace(
        probe_cs=tuple(float(value) for value in source_config["probe_cs"]),
        probe_max_iter=int(source_config["probe_max_iter"]),
    )

    fitted: list[FittedExtensionProbe] = []
    for family in FAMILIES:
        print(f"Extracting {family} training and validation features...", flush=True)
        model = _model_for_family(family, config)
        train_features, extracted_train_labels = extract_features(
            model, train_loader, device
        )
        val_features, extracted_val_labels = extract_features(model, val_loader, device)
        if not np.array_equal(extracted_train_labels, expected_train_labels):
            raise RuntimeError("Probe-training features do not match the saved split.")
        if not np.array_equal(extracted_val_labels, expected_val_labels):
            raise RuntimeError("Validation features do not match the saved split.")

        for layer, feature_matrix in train_features.items():
            original_feature_count = feature_matrix.shape[1]
            for projection_seed in config.projection_seeds:
                projected_train = random_project(
                    feature_matrix, config.controlled_dim, projection_seed
                )
                projected_val = random_project(
                    val_features[layer], config.controlled_dim, projection_seed
                )
                for task in tasks:
                    val_mask, val_targets = task_targets(extracted_val_labels, task)
                    for repeat in range(config.repeats):
                        sampling_seed = config.seed + repeat
                        for sample_size in config.sample_sizes:
                            positions = samples[(repeat, task.name, sample_size)]
                            selected_original = extracted_train_labels[positions]
                            mask, targets = task_targets(selected_original, task)
                            if not mask.all() or len(targets) != sample_size:
                                raise RuntimeError("Task sampling contains unrelated rows.")
                            counts = np.bincount(
                                targets, minlength=len(task.groups)
                            )
                            if counts.max() - counts.min() > 1:
                                raise RuntimeError("Task sampling is not near-balanced.")
                            selected = fit_probe(
                                projected_train[positions].copy(),
                                targets,
                                projected_val[val_mask].copy(),
                                val_targets,
                                fit_config,
                                sampling_seed,
                                len(task.groups),
                            )
                            fitted.append(
                                FittedExtensionProbe(
                                    repeat=repeat,
                                    sampling_seed=sampling_seed,
                                    task=task,
                                    sample_size=sample_size,
                                    projection_seed=projection_seed,
                                    family=family,
                                    layer=layer,
                                    original_feature_count=original_feature_count,
                                    training_indices=pool_ids[positions].copy(),
                                    training_class_counts=counts.copy(),
                                    selected=selected,
                                )
                            )
                del projected_train, projected_val
        del train_features, val_features, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return fitted, samples, pool_ids


def probe_manifest(
    fitted: Sequence[FittedExtensionProbe], source_seed: int
) -> list[dict[str, Any]]:
    """Serialize pre-test selection metadata for every fitted probe."""
    return [
        {
            "source_seed": source_seed,
            "repeat": run.repeat + 1,
            "sampling_seed": run.sampling_seed,
            "projection_seed": run.projection_seed,
            "task": run.task.name,
            "task_category": run.task.category,
            "sample_size": run.sample_size,
            "condition": run.condition,
            "family": run.family,
            "layer": run.layer,
            "training_class_counts": run.training_class_counts,
            "original_feature_count": run.original_feature_count,
            "feature_count": run.selected["classifier"].coef_.shape[1],
            "selected_c": run.selected["selected_c"],
            "validation_accuracy": run.selected["validation"]["accuracy"],
            "validation_macro_f1": run.selected["validation"]["macro_f1"],
        }
        for run in fitted
    ]


def save_probe_bundles(
    fitted: Sequence[FittedExtensionProbe], output_dir: Path, source_seed: int
) -> list[dict[str, Any]]:
    """Save compressed fitted probes grouped by task and representation."""
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in fitted:
        bundles[(run.task.name, run.condition)].append(
            {
                "source_seed": source_seed,
                "repeat": run.repeat + 1,
                "sampling_seed": run.sampling_seed,
                "projection_seed": run.projection_seed,
                "task": run.task.name,
                "task_category": run.task.category,
                "class_names": run.task.class_names,
                "original_label_groups": run.task.groups,
                "sample_size": run.sample_size,
                "condition": run.condition,
                "family": run.family,
                "layer": run.layer,
                "training_indices": run.training_indices,
                "training_class_counts": run.training_class_counts,
                "original_feature_count": run.original_feature_count,
                "feature_count": run.selected["classifier"].coef_.shape[1],
                "selected_c": run.selected["selected_c"],
                "validation": run.selected["validation"],
                "c_search": run.selected["c_search"],
                "scaler": run.selected["scaler"],
                "classifier": run.selected["classifier"],
            }
        )
    index = []
    for (task, condition), payload in sorted(bundles.items()):
        filename = f"{task}__{condition}.joblib"
        joblib.dump(payload, probe_dir / filename, compress=3)
        index.append(
            {
                "task": task,
                "condition": condition,
                "file": str(Path("probes") / filename),
                "probe_count": len(payload),
            }
        )
    return index


def _result_row(
    run: FittedExtensionProbe,
    scores: dict[str, Any],
    test_targets: np.ndarray,
    source_seed: int,
) -> dict[str, Any]:
    """Combine one fitted-probe identity with its held-out test metrics."""
    return {
        "source_seed": source_seed,
        "repeat": run.repeat + 1,
        "sampling_seed": run.sampling_seed,
        "projection_seed": run.projection_seed,
        "task": run.task.name,
        "task_category": run.task.category,
        "sample_size": run.sample_size,
        "condition": run.condition,
        "family": run.family,
        "layer": run.layer,
        "training_class_counts": run.training_class_counts.tolist(),
        "original_feature_count": run.original_feature_count,
        "feature_count": run.selected["classifier"].coef_.shape[1],
        "selected_c": run.selected["selected_c"],
        "validation_accuracy": run.selected["validation"]["accuracy"],
        "validation_macro_f1": run.selected["validation"]["macro_f1"],
        "n_test": len(test_targets),
        "test_class_counts": np.bincount(
            test_targets, minlength=len(run.task.groups)
        ).tolist(),
        **{f"test_{key}": value for key, value in scores.items()},
    }


def evaluate_fitted_probes(
    fitted: Sequence[FittedExtensionProbe],
    test_dataset: Dataset,
    test_indices: np.ndarray,
    tasks: list[Task],
    config: ExtensionProbeConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate already-fixed probes; this function performs no fitting."""
    loader = _loader(
        test_dataset, test_indices, config, device, config.seed + 1002
    )
    expected_labels = np.asarray(test_dataset.targets, dtype=np.int64)[test_indices]
    by_family: dict[str, list[FittedExtensionProbe]] = defaultdict(list)
    for run in fitted:
        by_family[run.family].append(run)
    rows = []
    for family in FAMILIES:
        print(f"Evaluating frozen {family} probes on the test split...", flush=True)
        model = _model_for_family(family, config)
        features, extracted_labels = extract_features(model, loader, device)
        if not np.array_equal(extracted_labels, expected_labels):
            raise RuntimeError("Test features do not match the saved split.")
        projected = {
            (layer, projection_seed): random_project(
                matrix, config.controlled_dim, projection_seed
            )
            for layer, matrix in features.items()
            for projection_seed in config.projection_seeds
        }
        for run in by_family[family]:
            task_mask, test_targets = task_targets(extracted_labels, run.task)
            matrix = projected[(run.layer, run.projection_seed)][task_mask]
            inputs = run.selected["scaler"].transform(matrix.copy())
            predictions = run.selected["classifier"].predict(inputs)
            scores = classification_metrics(
                test_targets, predictions, len(run.task.groups)
            )
            rows.append(_result_row(run, scores, test_targets, config.source_seed))
        del features, projected, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Add one deterministic constant baseline for every matched repeat and
    # projection. It predicts the majority target in that training sample.
    task_lookup = {task.name: task for task in tasks}
    reference = [run for run in fitted if run.family == "pixels"]
    for run in reference:
        task = task_lookup[run.task.name]
        task_mask, test_targets = task_targets(expected_labels, task)
        del task_mask
        majority = int(np.argmax(run.training_class_counts))
        scores = classification_metrics(
            test_targets,
            np.full_like(test_targets, majority),
            len(task.groups),
        )
        row = _result_row(run, scores, test_targets, config.source_seed)
        row.update(
            {
                "condition": "constant",
                "family": "constant",
                "layer": "constant",
                "original_feature_count": 0,
                "feature_count": 0,
                "selected_c": None,
                "validation_accuracy": None,
                "validation_macro_f1": None,
            }
        )
        rows.append(row)

    condition_order = {name: index for index, name in enumerate(CONDITION_ORDER)}
    task_order = {task.name: index for index, task in enumerate(tasks)}
    rows.sort(
        key=lambda row: (
            task_order[row["task"]],
            row["sample_size"],
            row["repeat"],
            row["projection_seed"],
            condition_order[row["condition"]],
        )
    )
    return rows


def aggregate_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate test metrics over sampling repeats and projection seeds."""
    keys = (
        "task",
        "task_category",
        "condition",
        "family",
        "layer",
        "sample_size",
        "original_feature_count",
        "feature_count",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = ("test_accuracy", "test_macro_f1", "test_balanced_accuracy")
    result = []
    for key, group in groups.items():
        record = dict(zip(keys, key))
        record["source_seed"] = group[0]["source_seed"]
        record["n_sampling_repeats"] = len({row["repeat"] for row in group})
        record["n_projection_seeds"] = len(
            {row["projection_seed"] for row in group}
        )
        record["n_runs"] = len(group)
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        selected_cs = [row["selected_c"] for row in group if row["selected_c"] is not None]
        if selected_cs:
            values, counts = np.unique(selected_cs, return_counts=True)
            record["selected_c_mode"] = float(values[np.argmax(counts)])
        else:
            record["selected_c_mode"] = None
        result.append(record)
    condition_order = {name: index for index, name in enumerate(CONDITION_ORDER)}
    result.sort(
        key=lambda row: (
            row["task"], condition_order[row["condition"]], row["sample_size"]
        )
    )
    return result


def paired_differences(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute fine-minus-random and coarse-minus-random matched differences."""
    keys = (
        "source_seed",
        "repeat",
        "sampling_seed",
        "projection_seed",
        "task",
        "sample_size",
        "layer",
    )
    references = {
        tuple(row[key] for key in keys): row
        for row in rows
        if row["family"] == "random"
    }
    result = []
    for row in rows:
        if row["family"] not in ("fine", "coarse"):
            continue
        reference = references[tuple(row[key] for key in keys)]
        difference = {key: row[key] for key in keys}
        difference.update(
            {
                "task_category": row["task_category"],
                "family": row["family"],
                "condition": row["condition"],
                "original_feature_count": row["original_feature_count"],
                "feature_count": row["feature_count"],
                "comparison": f"{row['family']}_minus_random",
            }
        )
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            difference[f"test_{metric}"] = (
                row[f"test_{metric}"] - reference[f"test_{metric}"]
            )
        result.append(difference)
    return result


def aggregate_paired_differences(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate matched trained-minus-random differences."""
    keys = (
        "task",
        "task_category",
        "comparison",
        "family",
        "layer",
        "sample_size",
        "original_feature_count",
        "feature_count",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in groups.items():
        record = dict(zip(keys, key))
        record["source_seed"] = group[0]["source_seed"]
        record["n_sampling_repeats"] = len({row["repeat"] for row in group})
        record["n_projection_seeds"] = len(
            {row["projection_seed"] for row in group}
        )
        record["n_pairs"] = len(group)
        for metric in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        result.append(record)
    result.sort(
        key=lambda row: (
            row["task"], row["comparison"], row["layer"], row["sample_size"]
        )
    )
    return result


def _plot_displays(layer: str) -> tuple[tuple[str, str, str, str, Any], ...]:
    """Return condition, label, color, marker, and line style definitions."""
    return (
        ("constant", "Constant", BASELINE_COLOR, "s", (0, (3, 2))),
        ("pixels", "Pixels", PIXEL_COLOR, "P", (0, (3, 2))),
        (f"random_{layer}", "Random CNN", RANDOM_COLOR, "o", (0, (1, 2))),
        (f"fine_{layer}", "Ten-class CNN", TRAINED_COLOR, "D", "-"),
        (f"coarse_{layer}", "Coarse CNN", COARSE_COLOR, "^", "-"),
    )


def _plot_panel(
    axis: plt.Axes,
    rows: Sequence[dict[str, Any]],
    layer: str,
    show_ylabel: bool,
) -> None:
    """Draw one layer's low-data curves using the shared minimal style."""
    sizes = sorted({int(row["sample_size"]) for row in rows})
    for condition, label, color, marker, linestyle in _plot_displays(layer):
        selected = sorted(
            [row for row in rows if row["condition"] == condition],
            key=lambda row: row["sample_size"],
        )
        if not selected:
            continue
        x = np.asarray([row["sample_size"] for row in selected])
        mean = np.asarray([row["test_macro_f1_mean"] for row in selected])
        sd = np.asarray([row["test_macro_f1_std"] for row in selected])
        axis.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markersize=4.4,
            linewidth=1.8,
            linestyle=linestyle,
            label=label,
        )
        axis.fill_between(
            x,
            np.maximum(mean - sd, 0),
            np.minimum(mean + sd, 1),
            color=color,
            alpha=0.09,
            linewidth=0,
        )
    lower = min(
        float(row["test_macro_f1_mean"]) - float(row["test_macro_f1_std"])
        for row in rows
    )
    y_min = max(0.0, np.floor((lower - 0.025) * 20) / 20)
    axis.set(
        title=layer.upper(),
        xlabel="Probe-training examples (total)",
        xscale="log",
        ylim=(y_min, 1.005),
    )
    if show_ylabel:
        axis.set_ylabel("Held-out test macro-F1")
    axis.set_xticks(sizes)
    axis.get_xaxis().set_major_formatter(ScalarFormatter())
    axis.get_xaxis().set_minor_locator(NullLocator())
    _style_axis(axis)


def plot_task_curves(
    aggregate: Sequence[dict[str, Any]], task: Task, directory: Path
) -> None:
    """Save one compact three-layer sample-efficiency figure for a task."""
    rows = [row for row in aggregate if row["task"] == task.name]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.8, 3.8),
        sharex=True,
        facecolor=PAPER,
        constrained_layout=True,
    )
    for index, (axis, layer) in enumerate(zip(axes, LAYERS)):
        _plot_panel(axis, rows, layer, show_ylabel=index == 0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        frameon=False,
        fontsize=8.5,
    )
    title = " · ".join(task.class_names)
    figure.suptitle(f"{title} · 32D controlled probes", color=INK, fontsize=13, y=1.13)
    first = rows[0]
    figure.text(
        0.5,
        -0.03,
        (
            f"Mean ± 1 sample SD · {first['n_sampling_repeats']} nested sampling repeats × "
            f"{first['n_projection_seeds']} Gaussian projection seeds"
        ),
        ha="center",
        color=MUTED_INK,
        fontsize=8.5,
    )
    _save_figure(figure, directory / f"{task.name}_sample_efficiency.png")
    plt.close(figure)


def plot_fc_overview(
    aggregate: Sequence[dict[str, Any]], tasks: Sequence[Task], path: Path
) -> None:
    """Save an overview of all tasks for the compact FC representation."""
    columns = 2
    rows = math.ceil(len(tasks) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10.5, 3 * rows),
        sharex=True,
        facecolor=PAPER,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, task in zip(axes.ravel(), tasks):
        rows = [row for row in aggregate if row["task"] == task.name]
        _plot_panel(axis, rows, "fc", show_ylabel=True)
        axis.set_title(" vs ".join(task.class_names), fontsize=9.5, color=INK)
    for axis in axes.ravel()[len(tasks) :]:
        axis.axis("off")
    legend = [
        Line2D(
            [0], [0], color=color, marker=marker, linestyle=linestyle, label=label
        )
        for _, label, color, marker, linestyle in _plot_displays("fc")
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Add-on probe sample efficiency · FC layer · 32D control",
        color=INK,
        fontsize=15,
        y=1.05,
    )
    figure.text(
        0.5,
        -0.012,
        "Exact total label budgets · mean ± 1 sample SD across 50 matched runs",
        ha="center",
        color=MUTED_INK,
        fontsize=9,
    )
    _save_figure(figure, path)
    plt.close(figure)


def _save_sampling_plan(
    path: Path,
    samples: dict[tuple[int, str, int], np.ndarray],
    pool_ids: np.ndarray,
) -> None:
    arrays = {"feature_pool": pool_ids}
    arrays.update(
        {
            f"repeat_{repeat + 1:02d}__{task}__n_{size:03d}": pool_ids[positions]
            for (repeat, task, size), positions in samples.items()
        }
    )
    np.savez_compressed(path, **arrays)


def run(
    config: ExtensionProbeConfig,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> None:
    """Run the add-on low-data experiment with optional test fixtures."""
    if (train_dataset is None) != (test_dataset is None):
        raise ValueError("Provide both dataset fixtures or neither.")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {config.output_dir}")
    source_config, splits = _load_source(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.output_dir / "config.json", config)
    seed_dir = config.source_dir / f"seed_{config.source_seed}"
    save_json(
        config.output_dir / "source_manifest.json",
        {
            "source_dir": config.source_dir.resolve(),
            "source_seed": config.source_seed,
            "coarse_target": source_config.get("coarse_target", "footwear"),
            "split_indices_sha256": _sha256(config.source_dir / "split_indices.npz"),
            "checkpoint_sha256": {
                family: _sha256(seed_dir / f"{family}_cnn.pt")
                for family in ("random", "fine", "coarse")
            },
            "source_probe_results_used": False,
            "cnn_retrained": False,
            "probe_cs": source_config["probe_cs"],
            "probe_max_iter": source_config["probe_max_iter"],
            "controlled_dim": config.controlled_dim,
            "projection_method": "GaussianRandomProjection",
            "projection_seeds": config.projection_seeds,
            "sample_size_unit": "total task-specific probe-training examples",
        },
    )
    coarse_target = source_config.get("coarse_target", "footwear")
    tasks = make_tasks(False, coarse_target)
    save_json(config.output_dir / "tasks.json", [asdict(task) for task in tasks])
    device = choose_device(config.requested_device)
    torch.set_num_threads(config.threads)
    started = time.perf_counter()
    save_json(
        config.output_dir / "status.json",
        {"stage": "preparing", "device": str(device)},
    )
    try:
        if train_dataset is None:
            from torchvision import datasets, transforms

            data_dir = config.data_dir or Path(source_config.get("data_dir", "data"))
            train_dataset = datasets.FashionMNIST(
                data_dir,
                train=True,
                transform=transforms.ToTensor(),
                download=False,
            )
        train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
        if np.any(splits["cnn_train"] >= len(train_labels)) or np.any(
            splits["probe_validation"] >= len(train_labels)
        ):
            raise ValueError("Saved training split indices exceed the dataset.")
        save_json(
            config.output_dir / "status.json",
            {"stage": "fitting", "device": str(device)},
        )
        with threadpool_limits(limits=config.threads):
            fitted, samples, pool_ids = fit_all_probes(
                train_dataset,
                train_labels,
                splits,
                tasks,
                source_config,
                config,
                device,
            )
            _save_sampling_plan(
                config.output_dir / "sample_indices.npz", samples, pool_ids
            )
            manifest = probe_manifest(fitted, config.source_seed)
            save_json(config.output_dir / "probe_manifest.json", manifest)
            bundle_index = save_probe_bundles(
                fitted, config.output_dir, config.source_seed
            )
            save_json(config.output_dir / "probe_bundles.json", bundle_index)

            # The official test dataset is not constructed until all selected
            # estimators and their metadata are durable on disk.
            if test_dataset is None:
                from torchvision import datasets, transforms

                data_dir = config.data_dir or Path(source_config.get("data_dir", "data"))
                test_dataset = datasets.FashionMNIST(
                    data_dir,
                    train=False,
                    transform=transforms.ToTensor(),
                    download=False,
                )
            if np.any(splits["test"] >= len(test_dataset)):
                raise ValueError("Saved test split indices exceed the dataset.")
            save_json(
                config.output_dir / "status.json",
                {"stage": "evaluating", "device": str(device)},
            )
            raw = evaluate_fitted_probes(
                fitted,
                test_dataset,
                splits["test"],
                tasks,
                config,
                device,
            )

        aggregate = aggregate_results(raw)
        paired = paired_differences(raw)
        paired_aggregate = aggregate_paired_differences(paired)
        save_csv(config.output_dir / "extension_probe_raw.csv", raw)
        save_json(config.output_dir / "extension_probe_raw.json", raw)
        save_csv(config.output_dir / "extension_probe_aggregate.csv", aggregate)
        save_json(config.output_dir / "extension_probe_aggregate.json", aggregate)
        save_csv(config.output_dir / "paired_differences_raw.csv", paired)
        save_csv(
            config.output_dir / "paired_differences_aggregate.csv", paired_aggregate
        )
        figures = config.output_dir / "figures"
        figures.mkdir()
        for task in tasks:
            plot_task_curves(aggregate, task, figures)
        plot_fc_overview(aggregate, tasks, figures / "fc_overview.png")
        elapsed = time.perf_counter() - started
        save_json(
            config.output_dir / "status.json",
            {
                "stage": "complete",
                "seconds": elapsed,
                "device": str(device),
                "source_seed": config.source_seed,
                "coarse_target": coarse_target,
                "fitted_probes": len(fitted),
                "probe_rows": len(raw),
                "aggregate_rows": len(aggregate),
                "paired_rows": len(paired),
                "paired_aggregate_rows": len(paired_aggregate),
                "controlled_dim": config.controlled_dim,
                "sampling_repeats": config.repeats,
                "projection_seeds": len(config.projection_seeds),
                "probe_bundles": len(bundle_index),
                "cnn_retrained": False,
            },
        )
        print(
            f"Finished in {elapsed:.1f} seconds: {config.output_dir.resolve()}",
            flush=True,
        )
    except Exception as error:
        save_json(
            config.output_dir / "status.json",
            {"stage": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    """Command-line entry point."""
    run(parse_arguments())


__all__ = [
    "CONDITION_ORDER",
    "DEFAULT_CONTROLLED_DIM",
    "DEFAULT_PROJECTION_SEEDS",
    "DEFAULT_REPEATS",
    "DEFAULT_SAMPLE_SIZES",
    "ExtensionProbeConfig",
    "FittedExtensionProbe",
    "aggregate_paired_differences",
    "aggregate_results",
    "evaluate_fitted_probes",
    "fit_all_probes",
    "main",
    "nested_task_samples",
    "paired_differences",
    "parse_arguments",
    "plot_fc_overview",
    "plot_task_curves",
    "probe_manifest",
    "run",
    "save_probe_bundles",
    "validate_config",
]
