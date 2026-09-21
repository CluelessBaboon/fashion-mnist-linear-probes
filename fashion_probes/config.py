"""Command-line configuration and shared experiment constants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


FOOTWEAR_CLASSES = frozenset({5, 7, 9})
TOP_CLASSES = frozenset({0, 2, 4, 6})
LAYER_NAMES = ("conv1", "conv2", "fc")

TRAIN_DATASET_SIZE = 60_000
TEST_DATASET_SIZE = 10_000

IMAGE_CHANNELS = 1
IMAGE_SIZE = 28
NUM_CLASSES = 10
CONV1_CHANNELS = 16
CONV2_CHANNELS = 32
CONV_KERNEL_SIZE = 3
CONV_PADDING = 1
POOL_SIZE = 2
HIDDEN_UNITS = 128

DEFAULT_PROBE_CS = (0.01, 0.1, 1.0)


@dataclass(frozen=True)
class Config:
    """Settings for one complete experiment.

    Attributes:
        data_dir: Directory used to download and cache Fashion-MNIST.
        output_dir: Directory receiving checkpoints, metrics, and figures.
        seed: Seed controlling initialization and data sampling.
        epochs: Number of CNN training epochs.
        batch_size: CNN training batch size.
        feature_batch_size: Evaluation and feature-extraction batch size.
        learning_rate: Adam learning rate.
        cnn_train_size: Number of original training examples used by the CNN.
        cnn_val_size: Number of examples reserved for CNN validation.
        probe_train_size: Probe subset size within the CNN training split.
        probe_val_size: Probe subset size within the CNN validation split.
        test_size: Test subset size, or None for the complete official test set.
        probe_cs: Candidate inverse L2 regularization strengths.
        probe_max_iter: Initial iteration limit for logistic regression.
        num_workers: Number of data-loader worker processes.
        requested_device: Requested device: auto, cpu, cuda, or mps.
        quick: Whether the reduced smoke-test defaults are active.
    """

    data_dir: Path
    output_dir: Path
    seed: int
    epochs: int
    batch_size: int
    feature_batch_size: int
    learning_rate: float
    cnn_train_size: int
    cnn_val_size: int
    probe_train_size: int
    probe_val_size: int
    test_size: int | None
    probe_cs: tuple[float, ...] | list[float]
    probe_max_iter: int
    num_workers: int
    requested_device: str
    quick: bool


def parse_arguments() -> Config:
    """Read command-line arguments and apply the original experiment defaults.

    Explicit arguments override the corresponding quick or full-run defaults.

    Returns:
        Validated configuration for one experiment.

    Raises:
        SystemExit: If argparse encounters an invalid command-line argument.
        ValueError: If the resulting configuration fails validation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a Fashion-MNIST CNN and probe its hidden representations "
            "for the broader footwear category."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fashion_mnist_probes"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cnn-train-size", type=int, default=None)
    parser.add_argument("--cnn-val-size", type=int, default=None)
    parser.add_argument("--probe-train-size", type=int, default=None)
    parser.add_argument("--probe-val-size", type=int, default=None)
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Use a stratified test subset. By default the full test set is used.",
    )
    parser.add_argument(
        "--probe-cs",
        type=float,
        nargs="+",
        default=DEFAULT_PROBE_CS,
        help="Candidate inverse regularization strengths for each probe.",
    )
    parser.add_argument("--probe-max-iter", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        dest="requested_device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller one-epoch end-to-end smoke test.",
    )
    args = parser.parse_args()

    if args.quick:
        defaults = {
            "epochs": 1,
            "cnn_train_size": 5_000,
            "cnn_val_size": 1_000,
            "probe_train_size": 1_000,
            "probe_val_size": 500,
            "test_size": 1_000,
        }
    else:
        defaults = {
            "epochs": 10,
            "cnn_train_size": 50_000,
            "cnn_val_size": 10_000,
            "probe_train_size": 5_000,
            "probe_val_size": 2_000,
            "test_size": None,
        }

    values = vars(args)
    for name, default_value in defaults.items():
        if values[name] is None:
            values[name] = default_value

    config = Config(**values)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Apply the original size and positivity checks.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If a checked value is nonpositive or a requested split
            exceeds its available source data.
    """
    positive_integer_fields = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "feature_batch_size": config.feature_batch_size,
        "cnn_train_size": config.cnn_train_size,
        "cnn_val_size": config.cnn_val_size,
        "probe_train_size": config.probe_train_size,
        "probe_val_size": config.probe_val_size,
        "probe_max_iter": config.probe_max_iter,
    }
    for name, value in positive_integer_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, received {value}.")

    if config.test_size is not None and config.test_size <= 0:
        raise ValueError("test_size must be positive when supplied.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not config.probe_cs or any(value <= 0 for value in config.probe_cs):
        raise ValueError("Every value in probe_cs must be positive.")
    if config.probe_train_size > config.cnn_train_size:
        raise ValueError("probe_train_size cannot exceed cnn_train_size.")
    if config.probe_val_size > config.cnn_val_size:
        raise ValueError("probe_val_size cannot exceed cnn_val_size.")
    if config.cnn_train_size + config.cnn_val_size > TRAIN_DATASET_SIZE:
        raise ValueError("Fashion-MNIST contains only 60,000 training images.")
    if config.test_size is not None and config.test_size > TEST_DATASET_SIZE:
        raise ValueError("Fashion-MNIST contains only 10,000 test images.")
