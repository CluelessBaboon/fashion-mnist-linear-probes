"""Orchestrate the original experiment in its original execution order."""

from __future__ import annotations

import copy
import gc
import random
from dataclasses import asdict

import numpy as np
import torch

from .config import LAYER_NAMES, parse_arguments
from .data import collect_footwear_labels, prepare_data
from .models import (
    FeatureCondition,
    SmallCNN,
    freeze_model,
    state_dict_on_cpu,
)
from .probes import (
    SelectedProbe,
    evaluate_selected_probes,
    extract_features,
    fit_select_probe,
    majority_baseline_row,
)
from .reporting import (
    save_confusion_matrices,
    save_learning_curves,
    save_probe_comparison,
    save_probe_objects,
    save_results_csv,
    write_json,
)
from .training import evaluate_cnn, train_cnn


def set_random_seeds(seed: int) -> None:
    """Seed supported random generators and request deterministic operations.

    Args:
        seed: Seed used by Python, NumPy, and PyTorch.

    Notes:
        Unsupported deterministic operations produce warnings, preserving
        the original behavior. Reproducibility across different hardware or
        software versions is not guaranteed by seeding alone.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested_device: str) -> torch.device:
    """Resolve an explicit device or select CUDA, MPS, then CPU.

    Args:
        requested_device: Device name; auto enables availability-based choice.

    Returns:
        Selected PyTorch device.

    Raises:
        RuntimeError: If explicitly requested CUDA or MPS is unavailable.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU."
        )
    if requested_device == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not mps_available:
            raise RuntimeError("MPS was requested, but it is not available.")
    return torch.device(requested_device)


def main() -> None:
    """Run CNN training, probe selection, final evaluation, and result export.

    Configuration is read from the command line. All probe fitting finishes
    before test metrics are computed. Existing output filenames are preserved;
    as in the original script, files in an existing output directory may be
    overwritten.

    Raises:
        ValueError: If configuration or data inputs are invalid.
        RuntimeError: If device selection or model fitting fails.
        OSError: If required files cannot be read or written.
    """
    config = parse_arguments()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", asdict(config))

    set_random_seeds(config.seed)
    device = resolve_device(config.requested_device)
    print(f"Using device: {device}")
    print("Preparing Fashion-MNIST data...")
    data = prepare_data(config, device)

    # Copy the same initialization before either representation is trained.
    trained_cnn = SmallCNN().to(device)
    random_cnn = copy.deepcopy(trained_cnn).to(device)
    torch.save(
        state_dict_on_cpu(random_cnn),
        config.output_dir / "random_cnn.pt",
    )
    freeze_model(random_cnn)

    print("Training the ten-class CNN...")
    history = train_cnn(
        trained_cnn,
        data.cnn_train,
        data.cnn_validation,
        config,
        device,
    )
    freeze_model(trained_cnn)
    write_json(config.output_dir / "cnn_history.json", history)
    save_learning_curves(history, config.output_dir)

    conditions = [FeatureCondition("pixels", "pixels", None)]
    for family, model in (("random", random_cnn), ("trained", trained_cnn)):
        for layer in LAYER_NAMES:
            conditions.append(FeatureCondition(family, layer, model))

    selected_probes: list[SelectedProbe] = []
    print("Fitting linear probes (CNN weights remain frozen)...")

    for condition in conditions:
        print(f"  Extracting {condition.name} features...")
        train_features, train_labels = extract_features(
            condition, data.probe_train, device
        )
        val_features, val_labels = extract_features(
            condition, data.probe_validation, device
        )
        selected = fit_select_probe(
            condition,
            train_features,
            train_labels,
            val_features,
            val_labels,
            config,
        )
        selected_probes.append(selected)
        print(
            f"  Selected C={selected.selected_c:g}; "
            f"validation macro-F1={selected.validation_macro_f1:.4f}"
        )

        del train_features, train_labels, val_features, val_labels
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_probe_objects(selected_probes, config.output_dir)

    # Test examples have not been used for optimization or model selection.
    print("Evaluating the frozen CNN and selected probes on test data...")
    cnn_test_metrics = evaluate_cnn(trained_cnn, data.test, device)
    probe_rows = evaluate_selected_probes(selected_probes, data.test, device)
    test_labels = collect_footwear_labels(data.test)
    baseline = majority_baseline_row(test_labels)
    all_rows = probe_rows + [baseline]

    save_results_csv(all_rows, config.output_dir / "probe_results.csv")
    write_json(config.output_dir / "probe_results.json", all_rows)
    save_probe_comparison(all_rows, config.output_dir)
    save_confusion_matrices(all_rows, config.output_dir)

    summary = {
        "actual_device": str(device),
        "cnn_test_metrics": cnn_test_metrics,
        "probe_results": all_rows,
        "important_note": (
            "The footwear target is derived from the ten original labels, so it "
            "is indirectly supervised by CNN training. One seed describes one run."
        ),
    }
    write_json(config.output_dir / "run_summary.json", summary)

    print(
        "Finished. "
        f"CNN test accuracy={cnn_test_metrics['accuracy']:.4f}. "
        f"Results were saved to {config.output_dir.resolve()}"
    )