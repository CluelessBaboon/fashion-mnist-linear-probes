"""Binary-superclass sample-efficiency probes using frozen CNN checkpoints.

The module deliberately has two phases: every scaler, regularization choice,
and classifier is fixed from the saved probe-training/probe-validation split
before the official test split is evaluated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullLocator, ScalarFormatter
from sklearn.random_projection import GaussianRandomProjection
from threadpoolctl import threadpool_limits
from torch.utils.data import Dataset

from .config import LAYER_NAMES, NUM_CLASSES
from .data import labels_to_footwear, labels_to_tops, make_loader
from .models import FeatureCondition, SmallCNN, freeze_model
from .pipeline import resolve_device, set_random_seeds
from .probes import SelectedProbe, binary_metrics, extract_features, fit_select_probe
from .reporting import (
    GRID,
    INK,
    MUTED_INK,
    PAPER,
    PIXEL_COLOR,
    RANDOM_COLOR,
    TRAINED_COLOR,
    _save_figure,
    _style_axis,
    save_results_csv,
    write_json,
)


DEFAULT_SAMPLE_SIZES = (10, 20, 30, 50, 80, 110, 150, 200)
DEFAULT_REPEATS = 10
DEFAULT_PROJECTION_SEEDS = (0, 1, 2, 3, 4)
TARGET_NAMES = ("footwear", "tops")
TARGET_LABELS = {
    "footwear": "Footwear vs non-footwear",
    "tops": "Tops vs non-tops",
}
TARGET_MAPPERS = {
    "footwear": labels_to_footwear,
    "tops": labels_to_tops,
}
CONDITION_ORDER = (
    "pixels",
    "random_conv1",
    "random_conv2",
    "random_fc",
    "trained_conv1",
    "trained_conv2",
    "trained_fc",
)


@dataclass(frozen=True)
class SampleEfficiencyConfig:
    """Configuration for a frozen-backbone sample-efficiency experiment."""

    source_dir: Path
    output_dir: Path
    data_dir: Path | None
    seed: int
    sample_sizes: tuple[int, ...]
    repeats: int
    feature_batch_size: int
    num_workers: int
    requested_device: str
    threads: int
    controlled_dim: int | None = None
    projection_seeds: tuple[int, ...] = ()


@dataclass
class FittedProbeRun:
    """One fitted probe together with its repeat and sample-size identity."""

    repeat: int
    sampling_seed: int
    sample_size: int
    target: str
    projection_seed: int | None
    original_feature_count: int
    training_indices: np.ndarray
    selected: SelectedProbe


def parse_arguments(argv: Sequence[str] | None = None) -> SampleEfficiencyConfig:
    """Parse command-line settings while retaining the prespecified design."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure frozen Fashion-MNIST representation quality as the "
            "number of probe-training examples increases."
        )
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("results/main_full_seed42")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sample_efficiency")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Fashion-MNIST root; defaults to the source run's data_dir.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        dest="requested_device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads used by PyTorch and scikit-learn.",
    )
    parser.add_argument(
        "--controlled-dim",
        type=int,
        help=(
            "Project every representation to this many dimensions before "
            "probe fitting. Omit to retain the original dimensions."
        ),
    )
    parser.add_argument(
        "--projection-seeds",
        type=int,
        nargs="+",
        help=(
            "Random-projection seeds. With --controlled-dim, defaults to "
            f"{DEFAULT_PROJECTION_SEEDS}."
        ),
    )
    values = vars(parser.parse_args(argv))
    values["sample_sizes"] = tuple(values["sample_sizes"])
    supplied_projection_seeds = values["projection_seeds"]
    if values["controlled_dim"] is None:
        values["projection_seeds"] = (
            tuple(supplied_projection_seeds)
            if supplied_projection_seeds is not None
            else ()
        )
    else:
        values["projection_seeds"] = tuple(
            supplied_projection_seeds or DEFAULT_PROJECTION_SEEDS
        )
    config = SampleEfficiencyConfig(**values)
    validate_config(config)
    return config


def validate_config(config: SampleEfficiencyConfig) -> None:
    """Reject settings that violate exact ten-class stratification."""
    if not config.sample_sizes:
        raise ValueError("At least one sample size is required.")
    if tuple(sorted(set(config.sample_sizes))) != config.sample_sizes:
        raise ValueError("sample_sizes must be strictly increasing and unique.")
    if any(size < NUM_CLASSES or size % NUM_CLASSES for size in config.sample_sizes):
        raise ValueError("Every sample size must be a positive multiple of ten.")
    if config.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if config.feature_batch_size <= 0 or config.threads <= 0:
        raise ValueError("feature_batch_size and threads must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if config.controlled_dim is not None and config.controlled_dim <= 0:
        raise ValueError("controlled_dim must be positive when supplied.")
    if config.controlled_dim is None and config.projection_seeds:
        raise ValueError("projection_seeds require controlled_dim.")
    if config.controlled_dim is not None and not config.projection_seeds:
        raise ValueError("At least one projection seed is required.")
    if len(set(config.projection_seeds)) != len(config.projection_seeds):
        raise ValueError("projection_seeds must be unique.")
    try:
        if config.source_dir.resolve() == config.output_dir.resolve():
            raise ValueError("output_dir must differ from the validated source run.")
    except OSError:
        pass


def nested_stratified_samples(
    available_indices: np.ndarray,
    labels: np.ndarray,
    sample_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Draw reproducible, nested samples balanced over all ten classes.

    Returned values are original dataset indices. Repeat numbers are zero-based
    in this low-level mapping; exported result rows use one-based numbering.
    """
    available = np.asarray(available_indices)
    original_labels = np.asarray(labels)
    sizes = tuple(sample_sizes)
    if available.ndim != 1 or not np.issubdtype(available.dtype, np.integer):
        raise ValueError("available_indices must be a one-dimensional integer array.")
    if len(np.unique(available)) != len(available):
        raise ValueError("available_indices must not contain duplicates.")
    if np.any(available < 0) or np.any(available >= len(original_labels)):
        raise ValueError("available_indices contain an out-of-range dataset index.")
    if not sizes or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sample_sizes must be strictly increasing and unique.")
    if any(size < NUM_CLASSES or size % NUM_CLASSES for size in sizes):
        raise ValueError("Every sample size must be a positive multiple of ten.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    largest_per_class = sizes[-1] // NUM_CLASSES
    by_class = []
    for class_id in range(NUM_CLASSES):
        candidates = available[original_labels[available] == class_id]
        if len(candidates) < largest_per_class:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} eligible examples; "
                f"{largest_per_class} are required."
            )
        by_class.append(candidates)

    samples: dict[tuple[int, int], np.ndarray] = {}
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        permutations = [rng.permutation(candidates) for candidates in by_class]
        for size in sizes:
            per_class = size // NUM_CLASSES
            selected = np.concatenate(
                [class_ids[:per_class] for class_ids in permutations]
            )
            # Row order has no scientific meaning, but shuffling avoids blocks.
            samples[(repeat, size)] = rng.permutation(selected)
    return samples


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate validation and test metrics across repeated nested samples."""
    if not rows:
        return []
    keys = (
        "target",
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

    metrics = (
        "validation_accuracy",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_target_f1",
    )
    order = {name: position for position, name in enumerate(CONDITION_ORDER)}
    result = []
    for key, group in groups.items():
        record = dict(zip(keys, key))
        record["n_sampling_repeats"] = len(
            {row["repeat"] for row in group}
        )
        projection_seeds = {
            row["projection_seed"]
            for row in group
            if row["projection_seed"] is not None
        }
        record["n_projection_seeds"] = len(projection_seeds)
        record["n_runs"] = len(group)
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        c_values = np.asarray([row["selected_c"] for row in group], dtype=float)
        unique, counts = np.unique(c_values, return_counts=True)
        record["selected_c_mode"] = float(unique[np.argmax(counts)])
        record["selected_c_mean"] = float(c_values.mean())
        result.append(record)
    target_order = {name: position for position, name in enumerate(TARGET_NAMES)}
    result.sort(
        key=lambda row: (
            target_order[row["target"]],
            order[row["condition"]],
            row["sample_size"],
        )
    )
    return result


def _load_source_metadata(config: SampleEfficiencyConfig) -> tuple[dict, dict]:
    """Load and validate source settings and the saved split archive."""
    required = (
        config.source_dir / "config.json",
        config.source_dir / "split_indices.npz",
        config.source_dir / "random_cnn.pt",
        config.source_dir / "trained_cnn_best.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifacts: " + ", ".join(missing))
    with (config.source_dir / "config.json").open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    for field in ("probe_cs", "probe_max_iter"):
        if field not in source_config:
            raise ValueError(f"Source config is missing {field!r}.")
    with np.load(config.source_dir / "split_indices.npz") as archive:
        required_keys = {
            "cnn_train", "cnn_validation", "probe_train", "probe_validation", "test"
        }
        if not required_keys <= set(archive.files):
            raise ValueError("Source split archive does not contain all main splits.")
        splits = {name: np.asarray(archive[name]).copy() for name in required_keys}
    for name, indices in splits.items():
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Source split {name!r} must be a 1D integer array.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Source split {name!r} contains duplicate indices.")
    if not set(splits["probe_train"]) <= set(splits["cnn_train"]):
        raise ValueError("Source probe_train is not contained in cnn_train.")
    if not set(splits["probe_validation"]) <= set(splits["cnn_validation"]):
        raise ValueError("Source probe_validation is not contained in cnn_validation.")
    return source_config, splits


def _load_model(path: Path, device: torch.device) -> SmallCNN:
    """Load and freeze a tensor-only main-experiment checkpoint."""
    model = SmallCNN()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    freeze_model(model)
    return model.to(device)


def _conditions(source_dir: Path, device: torch.device) -> list[FeatureCondition]:
    """Reconstruct all seven representation conditions without CNN training."""
    random_model = _load_model(source_dir / "random_cnn.pt", device)
    trained_model = _load_model(source_dir / "trained_cnn_best.pt", device)
    conditions = [FeatureCondition("pixels", "pixels", None)]
    for family, model in (("random", random_model), ("trained", trained_model)):
        conditions.extend(FeatureCondition(family, layer, model) for layer in LAYER_NAMES)
    return conditions


def random_project(
    features: np.ndarray,
    controlled_dim: int,
    projection_seed: int,
) -> np.ndarray:
    """Apply a reproducible Gaussian projection defined without labels or data.

    The projection matrix depends only on the input dimension, requested output
    dimension, and seed. Consequently, random and trained representations from
    the same layer receive exactly the same projection matrix for a given seed.
    """
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if controlled_dim <= 0 or controlled_dim > features.shape[1]:
        raise ValueError(
            "controlled_dim must be between 1 and the original feature count."
        )
    projector = GaussianRandomProjection(
        n_components=controlled_dim,
        random_state=projection_seed,
    )
    # GaussianRandomProjection.fit uses only the number of input columns. A
    # synthetic row makes that independence from experimental examples explicit.
    projector.fit(np.zeros((1, features.shape[1]), dtype=np.float32))
    return projector.transform(features).astype(np.float32, copy=False)


def _projection_seeds(config: SampleEfficiencyConfig) -> tuple[int | None, ...]:
    """Return one no-op identity or every prespecified projection seed."""
    if config.controlled_dim is None:
        return (None,)
    return tuple(config.projection_seeds)


def _loader(
    dataset: Dataset,
    indices: np.ndarray,
    config: SampleEfficiencyConfig,
    device: torch.device,
    seed: int,
):
    return make_loader(
        dataset,
        indices,
        batch_size=config.feature_batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )


def fit_all_probes(
    train_dataset: Dataset,
    train_labels: np.ndarray,
    splits: dict[str, np.ndarray],
    source_config: dict[str, Any],
    config: SampleEfficiencyConfig,
    device: torch.device,
) -> tuple[list[FittedProbeRun], dict[tuple[int, int], np.ndarray]]:
    """Fit every repeated probe without receiving or accessing test data."""
    samples = nested_stratified_samples(
        splits["probe_train"],
        train_labels,
        config.sample_sizes,
        config.repeats,
        config.seed,
    )
    # Only materialize rows used by at least one repeat. Every row still comes
    # from the source run's fixed probe_train split.
    pool = np.unique(np.concatenate(list(samples.values())))
    row_for_index = {int(index): row for row, index in enumerate(pool)}
    pool_loader = _loader(train_dataset, pool, config, device, config.seed + 100)
    val_loader = _loader(
        train_dataset,
        splits["probe_validation"],
        config,
        device,
        config.seed + 101,
    )
    pool_original_labels = torch.as_tensor(train_labels[pool])
    val_original_labels = torch.as_tensor(
        train_labels[splits["probe_validation"]]
    )

    fitted: list[FittedProbeRun] = []
    for condition in _conditions(config.source_dir, device):
        print(f"Extracting and fitting {condition.name}...")
        train_features, extracted_pool_targets = extract_features(
            condition, pool_loader, device
        )
        val_features, extracted_val_targets = extract_features(
            condition, val_loader, device
        )
        if not np.array_equal(
            extracted_pool_targets, labels_to_footwear(pool_original_labels).numpy()
        ):
            raise RuntimeError("Probe-training feature rows do not match the source split.")
        if not np.array_equal(
            extracted_val_targets, labels_to_footwear(val_original_labels).numpy()
        ):
            raise RuntimeError("Validation feature rows do not match the source split.")

        original_feature_count = train_features.shape[1]
        for projection_seed in _projection_seeds(config):
            if projection_seed is None:
                projected_train = train_features
                projected_val = val_features
            else:
                projected_train = random_project(
                    train_features, config.controlled_dim, projection_seed
                )
                projected_val = random_project(
                    val_features, config.controlled_dim, projection_seed
                )
            for target in TARGET_NAMES:
                train_targets = TARGET_MAPPERS[target](pool_original_labels).numpy()
                val_targets = TARGET_MAPPERS[target](val_original_labels).numpy()
                for repeat in range(config.repeats):
                    fit_config = SimpleNamespace(
                        probe_cs=tuple(
                            float(value) for value in source_config["probe_cs"]
                        ),
                        probe_max_iter=int(source_config["probe_max_iter"]),
                        seed=config.seed + repeat,
                    )
                    for size in config.sample_sizes:
                        original_indices = samples[(repeat, size)]
                        positions = np.asarray(
                            [row_for_index[int(index)] for index in original_indices]
                        )
                        selected = fit_select_probe(
                            condition,
                            projected_train[positions].copy(),
                            train_targets[positions],
                            projected_val.copy(),
                            val_targets,
                            fit_config,
                        )
                        fitted.append(
                            FittedProbeRun(
                                repeat=repeat,
                                sampling_seed=config.seed + repeat,
                                sample_size=size,
                                target=target,
                                projection_seed=projection_seed,
                                original_feature_count=original_feature_count,
                                training_indices=original_indices.copy(),
                                selected=selected,
                            )
                        )
        del train_features, extracted_pool_targets, val_features, extracted_val_targets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return fitted, samples


def probe_manifest(fitted: list[FittedProbeRun]) -> list[dict[str, Any]]:
    """Create pre-test metadata proving that all probe choices are fixed."""
    rows = []
    for run in fitted:
        selected = run.selected
        rows.append(
            {
                "repeat": run.repeat + 1,
                "sampling_seed": run.sampling_seed,
                "sample_size": run.sample_size,
                "target": run.target,
                "projection_seed": run.projection_seed,
                "condition": selected.condition.name,
                "family": selected.condition.family,
                "layer": selected.condition.layer,
                "original_feature_count": run.original_feature_count,
                "feature_count": selected.feature_count,
                "selected_c": selected.selected_c,
                "validation_accuracy": selected.validation_accuracy,
                "validation_macro_f1": selected.validation_macro_f1,
                "training_indices": run.training_indices,
            }
        )
    return rows


def save_fitted_probe_bundles(
    fitted: list[FittedProbeRun],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Save compact fitted-probe bundles grouped by target and condition.

    CNN objects are deliberately excluded because the source checkpoints are
    already hash-addressed in ``source_manifest.json``. Each payload retains
    the fitted scaler and logistic classifier plus the complete run identity.
    """
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    bundles: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in fitted:
        selected = run.selected
        bundles[(run.target, selected.condition.name)].append(
            {
                "target": run.target,
                "condition": selected.condition.name,
                "family": selected.condition.family,
                "layer": selected.condition.layer,
                "repeat": run.repeat + 1,
                "sampling_seed": run.sampling_seed,
                "projection_seed": run.projection_seed,
                "sample_size": run.sample_size,
                "training_indices": run.training_indices,
                "original_feature_count": run.original_feature_count,
                "feature_count": selected.feature_count,
                "selected_c": selected.selected_c,
                "validation_accuracy": selected.validation_accuracy,
                "validation_macro_f1": selected.validation_macro_f1,
                "scaler": selected.scaler,
                "classifier": selected.classifier,
            }
        )

    index = []
    for (target, condition), payloads in sorted(bundles.items()):
        filename = f"{target}__{condition}.joblib"
        joblib.dump(payloads, probe_dir / filename, compress=3)
        index.append(
            {
                "target": target,
                "condition": condition,
                "file": str(Path("probes") / filename),
                "probe_count": len(payloads),
            }
        )
    return index


def evaluate_fitted_probes(
    fitted: list[FittedProbeRun],
    test_dataset: Dataset,
    test_indices: np.ndarray,
    config: SampleEfficiencyConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate already-fitted probes; this function performs no fitting."""
    loader = _loader(test_dataset, test_indices, config, device, config.seed + 102)
    by_condition: dict[str, list[FittedProbeRun]] = defaultdict(list)
    for run in fitted:
        by_condition[run.selected.condition.name].append(run)

    rows = []
    for condition_name in CONDITION_ORDER:
        condition_runs = by_condition[condition_name]
        if not condition_runs:
            continue
        condition = condition_runs[0].selected.condition
        test_features, extracted_test_targets = extract_features(
            condition, loader, device
        )
        original_test_labels = torch.as_tensor(
            np.asarray(test_dataset.targets, dtype=np.int64)[test_indices]
        )
        if not np.array_equal(
            extracted_test_targets,
            labels_to_footwear(original_test_labels).numpy(),
        ):
            raise RuntimeError("Test feature rows do not match the source split.")
        projected_test = {
            projection_seed: (
                test_features
                if projection_seed is None
                else random_project(
                    test_features, config.controlled_dim, projection_seed
                )
            )
            for projection_seed in _projection_seeds(config)
        }
        for run in condition_runs:
            selected = run.selected
            test_targets = TARGET_MAPPERS[run.target](original_test_labels).numpy()
            # StandardScaler(copy=False) belongs to the validated main method;
            # copy here so one repeat cannot mutate another repeat's test matrix.
            inputs = selected.scaler.transform(
                projected_test[run.projection_seed].copy()
            )
            probabilities = selected.classifier.predict_proba(inputs)[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int64)
            metrics = binary_metrics(test_targets, predictions)
            rows.append(
                {
                    "target": run.target,
                    "repeat": run.repeat + 1,
                    "sampling_seed": run.sampling_seed,
                    "projection_seed": run.projection_seed,
                    "sample_size": run.sample_size,
                    "examples_per_original_class": run.sample_size // NUM_CLASSES,
                    "training_class_counts": [run.sample_size // NUM_CLASSES] * NUM_CLASSES,
                    "condition": condition.name,
                    "family": condition.family,
                    "layer": condition.layer,
                    "original_feature_count": run.original_feature_count,
                    "feature_count": selected.feature_count,
                    "selected_c": selected.selected_c,
                    "validation_accuracy": selected.validation_accuracy,
                    "validation_macro_f1": selected.validation_macro_f1,
                    "n_test": len(test_targets),
                    "test_accuracy": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_target_f1": metrics["footwear_f1"],
                    "true_negative": metrics["true_negative"],
                    "false_positive": metrics["false_positive"],
                    "false_negative": metrics["false_negative"],
                    "true_positive": metrics["true_positive"],
                }
            )
        del test_features, extracted_test_targets, projected_test
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    order = {name: position for position, name in enumerate(CONDITION_ORDER)}
    target_order = {name: position for position, name in enumerate(TARGET_NAMES)}
    rows.sort(
        key=lambda row: (
            row["repeat"],
            -1 if row["projection_seed"] is None else row["projection_seed"],
            row["sample_size"],
            target_order[row["target"]],
            order[row["condition"]],
        )
    )
    return rows


def save_sample_efficiency_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot test macro-F1 means and SD across sampling/projection runs."""
    figure, axes = plt.subplots(
        len(TARGET_NAMES),
        len(LAYER_NAMES),
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        facecolor=PAPER,
    )
    colors = {
        "random": RANDOM_COLOR,
        "trained": TRAINED_COLOR,
        "pixels": PIXEL_COLOR,
    }
    markers = {"random": "o", "trained": "D", "pixels": "P"}
    all_lower_bounds = [
        float(row["test_macro_f1_mean"]) - float(row["test_macro_f1_std"])
        for row in rows
    ]
    y_min = max(0.0, np.floor((min(all_lower_bounds) - 0.025) * 20) / 20)
    for row_index, target in enumerate(TARGET_NAMES):
        for column_index, layer in enumerate(LAYER_NAMES):
            axis = axes[row_index, column_index]
            displays = (
                ("pixels", "Raw pixels", "pixels", (0, (3, 2))),
                (f"random_{layer}", "Random CNN", "random", (0, (1, 2))),
                (f"trained_{layer}", "Trained CNN", "trained", "-"),
            )
            for condition, label, color, style in displays:
                selected = sorted(
                    [
                        row
                        for row in rows
                        if row["target"] == target and row["condition"] == condition
                    ],
                    key=lambda row: row["sample_size"],
                )
                x = np.asarray([row["sample_size"] for row in selected])
                mean = np.asarray([row["test_macro_f1_mean"] for row in selected])
                sd = np.asarray([row["test_macro_f1_std"] for row in selected])
                axis.plot(
                    x,
                    mean,
                    marker=markers[color],
                    markersize=4.8,
                    color=colors[color],
                    linestyle=style,
                    linewidth=1.9,
                    label=label,
                )
                axis.fill_between(
                    x,
                    np.maximum(mean - sd, 0),
                    np.minimum(mean + sd, 1),
                    color=colors[color],
                    alpha=0.10,
                    linewidth=0,
                )
            axis.set(xscale="log", ylim=(y_min, 1.005))
            if row_index == 0:
                axis.set_title(layer.upper())
            if row_index == len(TARGET_NAMES) - 1:
                axis.set_xlabel("Probe-training examples (total)")
            if column_index == 0:
                axis.set_ylabel(f"{TARGET_LABELS[target]}\nTest macro-F1")
            axis.set_xticks(x)
            axis.get_xaxis().set_major_formatter(ScalarFormatter())
            axis.get_xaxis().set_minor_locator(NullLocator())
            _style_axis(axis)
            axis.grid(color=GRID, linewidth=0.7, alpha=0.7)
    feature_counts = {int(row["feature_count"]) for row in rows}
    original_counts = {int(row["original_feature_count"]) for row in rows}
    if len(feature_counts) == 1 and feature_counts != original_counts:
        budget = next(iter(feature_counts))
        title = (
            "Fashion-MNIST binary-superclass sample efficiency "
            f"({budget}D random-projection control)"
        )
    else:
        title = "Fashion-MNIST binary-superclass probe sample efficiency"
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(title, color=INK, fontsize=15, y=1.09)
    first_row = rows[0]
    figure.text(
        0.5,
        -0.015,
        (
            "Mean ± 1 sample SD across "
            f"{first_row['n_sampling_repeats']} sampling repeats × "
            f"{first_row['n_projection_seeds']} projection seeds"
        ),
        ha="center",
        color=MUTED_INK,
        fontsize=9,
    )
    _save_figure(figure, path)
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_sampling_plan(
    path: Path, samples: dict[tuple[int, int], np.ndarray]
) -> None:
    arrays = {
        f"repeat_{repeat + 1:02d}_n_{size:03d}": indices
        for (repeat, size), indices in samples.items()
    }
    np.savez_compressed(path, **arrays)


def run(
    config: SampleEfficiencyConfig,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
) -> None:
    """Run the frozen-checkpoint experiment, optionally with test fixtures."""
    if (train_dataset is None) != (test_dataset is None):
        raise ValueError("Provide both dataset fixtures or neither.")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {config.output_dir}")
    source_config, splits = _load_source_metadata(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", asdict(config))
    write_json(
        config.output_dir / "source_manifest.json",
        {
            "source_dir": config.source_dir.resolve(),
            "split_indices_sha256": _sha256(config.source_dir / "split_indices.npz"),
            "random_cnn_sha256": _sha256(config.source_dir / "random_cnn.pt"),
            "trained_cnn_sha256": _sha256(config.source_dir / "trained_cnn_best.pt"),
            "source_probe_cs": source_config["probe_cs"],
            "source_probe_max_iter": source_config["probe_max_iter"],
            "source_probe_results_used": False,
            "targets": TARGET_NAMES,
            "controlled_dim": config.controlled_dim,
            "projection_seeds": config.projection_seeds,
            "projection_method": (
                "GaussianRandomProjection"
                if config.controlled_dim is not None
                else None
            ),
            "cnn_retrained": False,
        },
    )
    device = resolve_device(config.requested_device)
    torch.set_num_threads(config.threads)
    set_random_seeds(config.seed)
    started = time.perf_counter()
    write_json(
        config.output_dir / "status.json",
        {"stage": "preparing", "device": str(device)},
    )
    try:
        if train_dataset is None:
            from torchvision import datasets, transforms

            data_dir = config.data_dir or Path(source_config.get("data_dir", "data"))
            transform = transforms.ToTensor()
            train_dataset = datasets.FashionMNIST(
                data_dir, train=True, transform=transform, download=False
            )
        train_labels = np.asarray(train_dataset.targets, dtype=np.int64)
        if np.any(splits["probe_train"] >= len(train_labels)) or np.any(
            splits["probe_validation"] >= len(train_labels)
        ):
            raise ValueError("Saved training split indices exceed the dataset.")

        write_json(
            config.output_dir / "status.json",
            {"stage": "fitting", "device": str(device)},
        )
        with threadpool_limits(limits=config.threads):
            fitted, samples = fit_all_probes(
                train_dataset, train_labels, splits, source_config, config, device
            )
            _save_sampling_plan(config.output_dir / "sample_indices.npz", samples)
            write_json(config.output_dir / "probe_manifest.json", probe_manifest(fitted))
            probe_bundle_index = save_fitted_probe_bundles(
                fitted,
                config.output_dir,
            )
            write_json(
                config.output_dir / "probe_bundles.json",
                probe_bundle_index,
            )

            # The official test dataset is first loaded/used only after every
            # probe has been selected and the pre-test manifest is durable.
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
            write_json(
                config.output_dir / "status.json",
                {"stage": "evaluating", "device": str(device)},
            )
            raw_rows = evaluate_fitted_probes(
                fitted, test_dataset, splits["test"], config, device
            )

        aggregate_rows = aggregate_results(raw_rows)
        save_results_csv(raw_rows, config.output_dir / "sample_efficiency_raw.csv")
        write_json(config.output_dir / "sample_efficiency_raw.json", raw_rows)
        save_results_csv(
            aggregate_rows, config.output_dir / "sample_efficiency_aggregate.csv"
        )
        write_json(
            config.output_dir / "sample_efficiency_aggregate.json", aggregate_rows
        )
        save_sample_efficiency_plot(
            aggregate_rows, config.output_dir / "sample_efficiency.png"
        )
        elapsed = time.perf_counter() - started
        write_json(
            config.output_dir / "status.json",
            {
                "stage": "complete",
                "seconds": elapsed,
                "device": str(device),
                "probe_rows": len(raw_rows),
                "aggregate_rows": len(aggregate_rows),
                "controlled_dim": config.controlled_dim,
                "projection_seeds": len(config.projection_seeds),
                "probe_bundles": len(probe_bundle_index),
                "cnn_retrained": False,
            },
        )
        print(f"Finished in {elapsed:.1f} seconds: {config.output_dir.resolve()}")
    except Exception as error:
        write_json(
            config.output_dir / "status.json",
            {"stage": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    """Command-line entry point."""
    run(parse_arguments())


__all__ = [
    "CONDITION_ORDER",
    "DEFAULT_PROJECTION_SEEDS",
    "DEFAULT_REPEATS",
    "DEFAULT_SAMPLE_SIZES",
    "TARGET_LABELS",
    "TARGET_MAPPERS",
    "TARGET_NAMES",
    "FittedProbeRun",
    "SampleEfficiencyConfig",
    "aggregate_results",
    "evaluate_fitted_probes",
    "fit_all_probes",
    "main",
    "nested_stratified_samples",
    "parse_arguments",
    "probe_manifest",
    "random_project",
    "run",
    "save_fitted_probe_bundles",
    "save_sample_efficiency_plot",
    "validate_config",
]
