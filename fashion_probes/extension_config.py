"""Configuration and explicit task definitions for the extension."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
CLASS_SLUGS = (
    "tshirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
)
FOOTWEAR = (5, 7, 9)
NON_FOOTWEAR = (0, 1, 2, 3, 4, 6, 8)
TOPS = (0, 2, 4, 6)
NON_TOPS = (1, 3, 5, 7, 8, 9)
COARSE_TARGETS = {
    "footwear": (FOOTWEAR, NON_FOOTWEAR),
    "tops": (TOPS, NON_TOPS),
}
LAYERS = ("conv1", "conv2", "fc")
FAMILIES = ("pixels", "random", "fine", "coarse")


@dataclass(frozen=True)
class Task:
    """Define a probe's class groups in target-label order.

    Attributes:
        name: Stable output identifier.
        category: within_<coarse target>, cross_pooled, or cross_pair.
        groups: Original labels mapping to targets 0, 1, and optionally 2.
        class_names: Human-readable names in the same target order.
    """

    name: str
    category: str
    groups: tuple[tuple[int, ...], ...]
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class ExtensionConfig:
    """Validated settings for the extension.

    Paths are relative to the working directory unless supplied as absolute
    paths. Budgets count examples per downstream target class. Seeds vary
    initialization and probe sampling; split_seed fixes the dataset split.
    CNN and probe validation sizes are independent of probe-training budgets.
    """

    data_dir: Path
    output_dir: Path
    coarse_target: str
    seeds: tuple[int, ...]
    split_seed: int
    epochs: int
    batch_size: int
    feature_batch_size: int
    learning_rate: float
    cnn_train_size: int
    cnn_val_size: int
    probe_val_size: int
    test_size: int | None
    probe_sizes: tuple[int, ...]
    probe_cs: tuple[float, ...]
    probe_max_iter: int
    num_workers: int
    threads: int
    device: str
    quick: bool
    include_cross_pairs: bool
    exclude_test_indices: tuple[Path, ...]


def parse_arguments(argv: list[str] | None = None) -> ExtensionConfig:
    """Parse settings; explicit values override quick/full defaults.

    Args:
        argv: Argument list, or None to read the command line.

    Returns:
        Validated experiment configuration.

    Raises:
        SystemExit: If syntax or numerical configuration is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/extension"))
    parser.add_argument(
        "--coarse-target",
        choices=tuple(COARSE_TARGETS),
        default="footwear",
        help="Semantic group used as the positive class for coarse CNN training.",
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", nargs="+", type=int)
    seeds.add_argument("--seed", type=int, help="Convenience option for one seed.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cnn-train-size", type=int)
    parser.add_argument("--cnn-val-size", type=int)
    parser.add_argument("--probe-val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument(
        "--probe-sizes",
        nargs="+",
        type=int,
        help="Training examples per downstream target class.",
    )
    parser.add_argument("--probe-cs", nargs="+", type=float)
    parser.add_argument("--probe-max-iter", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="CPU thread limit for Torch and numerical libraries.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--include-cross-pairs",
        action="store_true",
        help="Also run every positive-class versus negative-class pair.",
    )
    parser.add_argument(
        "--exclude-test-indices",
        type=Path,
        nargs="+",
        default=[],
        help="Previous split NPZ files; exclude the union of their test IDs.",
    )
    values = vars(parser.parse_args(argv))
    quick = values["quick"]
    defaults = {
        "epochs": 1 if quick else 10,
        "cnn_train_size": 5000 if quick else 50000,
        "cnn_val_size": 1000 if quick else 10000,
        "probe_val_size": 500 if quick else 2000,
        "test_size": 1000 if quick else None,
        "probe_sizes": (5, 10) if quick else (5, 10, 30, 100),
        "probe_cs": (0.1,) if quick else (0.01, 0.1, 1.0),
    }
    for name, default in defaults.items():
        if values[name] is None:
            values[name] = default
    seed = values.pop("seed")
    values["seeds"] = tuple(
        values["seeds"]
        or ([seed] if seed is not None else ([42] if quick else [42, 43, 44]))
    )
    values["probe_sizes"] = tuple(sorted(set(values["probe_sizes"])))
    values["probe_cs"] = tuple(sorted(set(values["probe_cs"])))
    values["exclude_test_indices"] = tuple(values["exclude_test_indices"])
    config = ExtensionConfig(**values)
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def validate_config(config: ExtensionConfig) -> None:
    """Reject infeasible settings before downloading data or creating results.

    Args:
        config: Settings to validate.

    Raises:
        ValueError: If budgets, split sizes, seeds, or numerical values are invalid.
    """
    for name in (
        "epochs",
        "batch_size",
        "feature_batch_size",
        "probe_max_iter",
        "threads",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    for name in ("cnn_train_size", "cnn_val_size", "probe_val_size"):
        if getattr(config, name) < 10:
            raise ValueError(
                f"{name} must allow at least one example of all 10 classes."
            )
    if config.cnn_train_size + config.cnn_val_size > 60000:
        raise ValueError("CNN training and validation sizes exceed 60000.")
    if config.probe_val_size > config.cnn_val_size:
        raise ValueError("probe_val_size exceeds cnn_val_size.")
    if config.test_size is not None and not 10 <= config.test_size <= 10000:
        raise ValueError("test_size must be between 10 and 10000.")
    if not config.probe_sizes or min(config.probe_sizes) < 1:
        raise ValueError("Probe budgets must be positive.")
    if max(config.probe_sizes) > config.cnn_train_size // 10:
        raise ValueError(
            "The largest budget exceeds training examples per original class."
        )
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not config.probe_cs or any(
        not math.isfinite(c) or c <= 0 for c in config.probe_cs
    ):
        raise ValueError("All C values must be finite and positive.")
    if len(config.seeds) != len(set(config.seeds)):
        raise ValueError("Seeds must be distinct.")
    if any(
        not 0 <= seed < 2**32 - 10000 for seed in (*config.seeds, config.split_seed)
    ):
        raise ValueError("Seeds must be nonnegative and below 2**32 - 10000.")


def make_tasks(
    include_cross_pairs: bool = False, coarse_target: str = "footwear"
) -> list[Task]:
    """Return prespecified probe tasks, without consulting any results.

    Args:
        include_cross_pairs: Add every individual cross-category contrast.
        coarse_target: Semantic grouping used by the binary coarse CNN.

    Returns:
        All within-group and pooled cross-category tasks for the target.

    Raises:
        ValueError: If coarse_target is unknown.
    """
    if coarse_target not in COARSE_TARGETS:
        raise ValueError(f"Unknown coarse target: {coarse_target!r}.")
    positive, negative = COARSE_TARGETS[coarse_target]
    group_size_name = {3: "three", 4: "four"}.get(len(positive), str(len(positive)))
    negative_name = f"Non-{coarse_target}"
    tasks = [
        Task(
            f"{coarse_target}_{group_size_name}_way",
            f"within_{coarse_target}",
            tuple((label,) for label in positive),
            tuple(CLASS_NAMES[label] for label in positive),
        )
    ]
    for left, right in combinations(positive, 2):
        tasks.append(
            Task(
                f"{CLASS_SLUGS[left]}_vs_{CLASS_SLUGS[right]}",
                f"within_{coarse_target}",
                ((left,), (right,)),
                (CLASS_NAMES[left], CLASS_NAMES[right]),
            )
        )
    for label in positive:
        tasks.append(
            Task(
                f"{CLASS_SLUGS[label]}_vs_non_{coarse_target}",
                "cross_pooled",
                (negative, (label,)),
                (negative_name, CLASS_NAMES[label]),
            )
        )
    if include_cross_pairs:
        for label, other in product(positive, negative):
            tasks.append(
                Task(
                    f"{CLASS_SLUGS[label]}_vs_{CLASS_SLUGS[other]}",
                    "cross_pair",
                    ((other,), (label,)),
                    (CLASS_NAMES[other], CLASS_NAMES[label]),
                )
            )
    return tasks
