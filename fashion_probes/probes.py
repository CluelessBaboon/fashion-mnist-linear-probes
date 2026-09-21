"""Frozen feature extraction, linear-probe fitting, and binary evaluation."""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config
from .data import labels_to_footwear
from .models import FeatureCondition


@dataclass
class SelectedProbe:
    """A validation-selected classifier and its training-fitted scaler.

    Attributes:
        condition: Representation used by the classifier.
        scaler: StandardScaler fitted only on probe-training features.
        classifier: Logistic-regression model selected by validation macro-F1.
        selected_c: Selected inverse regularization strength.
        validation_accuracy: Selected classifier's validation accuracy.
        validation_macro_f1: Selected classifier's validation macro-F1.
        feature_count: Number of input features.
    """

    condition: FeatureCondition
    scaler: StandardScaler
    classifier: LogisticRegression
    selected_c: float
    validation_accuracy: float
    validation_macro_f1: float
    feature_count: int


def extract_features(
    condition: FeatureCondition,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened representations and binary targets without gradients.

    Args:
        condition: Raw-pixel or hidden-layer representation to extract.
        loader: Nonempty loader yielding images and CPU original-label tensors.
        device: Device used for images and CNN inference.

    Returns:
        Float32 feature matrix of shape (examples, features) and corresponding
        one-dimensional binary target array.

    Raises:
        ValueError: If a hidden-layer condition has no CNN.
        KeyError: If the requested hidden layer is unavailable.

    Notes:
        A supplied CNN must already be on the specified device. It is left
        in evaluation mode.
    """
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    if condition.model is not None:
        condition.model.eval()

    with torch.no_grad():
        for images, original_labels in loader:
            footwear_labels = labels_to_footwear(original_labels)
            images = images.to(device, non_blocking=True)

            if condition.layer == "pixels":
                features = torch.flatten(images, start_dim=1)
            else:
                if condition.model is None:
                    raise ValueError(
                        "A CNN is required for a hidden-layer condition."
                    )
                _, activations = condition.model(images)
                if condition.layer not in activations:
                    raise KeyError(
                        f"Unknown activation layer: {condition.layer}"
                    )
                features = torch.flatten(
                    activations[condition.layer], start_dim=1
                )

            feature_batches.append(
                features.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            label_batches.append(footwear_labels.numpy())

    return np.concatenate(feature_batches), np.concatenate(label_batches)


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    """Fit L2 logistic regression, retrying once after a convergence warning.

    Args:
        features: Standardized training matrix with shape (examples, features).
        labels: Binary training targets.
        c_value: Positive inverse regularization strength.
        max_iter: Initial maximum number of optimization iterations.
        seed: Random-state value supplied to scikit-learn.

    Returns:
        Fitted classifier that produced no ConvergenceWarning.

    Raises:
        RuntimeError: If both fits produce convergence warnings.
        ValueError: If the supplied data or classifier settings are invalid.

    Notes:
        The retry starts a fresh fit with five times the original iteration
        budget, matching the original implementation.
    """
    for iteration_limit in (max_iter, max_iter * 5):
        probe = LogisticRegression(
            penalty="l2",
            C=c_value,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=iteration_limit,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            probe.fit(features, labels)

        converged = not any(
            issubclass(item.category, ConvergenceWarning)
            for item in caught_warnings
        )
        if converged:
            return probe

    raise RuntimeError(
        f"Logistic regression with C={c_value} did not converge after "
        f"{max_iter * 5} iterations. Increase --probe-max-iter."
    )


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Calculate the original binary classification metrics.

    Args:
        labels: True targets, with footwear encoded as 1.
        predictions: Predicted targets encoded as 0 or 1.

    Returns:
        Accuracy, macro-F1, footwear F1, and the four confusion-matrix counts.
    """
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "footwear_f1": float(
            f1_score(
                labels,
                predictions,
                pos_label=1,
                average="binary",
                zero_division=0,
            )
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def fit_select_probe(
    condition: FeatureCondition,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: Config,
) -> SelectedProbe:
    """Fit training-only scaling and select a probe by validation macro-F1.

    Args:
        condition: Feature representation being evaluated.
        train_features: Probe-training feature matrix.
        train_labels: Binary probe-training targets.
        val_features: Probe-validation feature matrix.
        val_labels: Binary probe-validation targets.
        config: Candidate regularization values and fitting settings.

    Returns:
        Selected classifier, scaler, and validation metadata.

    Raises:
        RuntimeError: If fitting fails to converge or no candidate is selected.
        ValueError: If input data or settings are incompatible.

    Notes:
        StandardScaler(copy=False) may modify the supplied feature arrays,
        preserving the original memory behavior. Candidates are considered
        in the supplied order; the first candidate wins a macro-F1 tie.
        The decision threshold remains 0.5.
    """
    scaler = StandardScaler(copy=False)
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    best_probe: LogisticRegression | None = None
    best_c: float | None = None
    best_metrics: dict[str, Any] | None = None

    for c_value in config.probe_cs:
        probe = fit_logistic_regression(
            train_features,
            train_labels,
            c_value=c_value,
            max_iter=config.probe_max_iter,
            seed=config.seed,
        )
        probabilities = probe.predict_proba(val_features)[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        metrics = binary_metrics(val_labels, predictions)

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_probe = probe
            best_c = c_value
            best_metrics = metrics

    if best_probe is None or best_c is None or best_metrics is None:
        raise RuntimeError(f"No probe was selected for {condition.name}.")

    return SelectedProbe(
        condition=condition,
        scaler=scaler,
        classifier=best_probe,
        selected_c=float(best_c),
        validation_accuracy=best_metrics["accuracy"],
        validation_macro_f1=best_metrics["macro_f1"],
        feature_count=train_features.shape[1],
    )


def evaluate_selected_probes(
    selected_probes: list[SelectedProbe],
    test_loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate already-selected probes using their fitted transformations.

    Args:
        selected_probes: Probes whose fitting and model selection are complete.
        test_loader: Nonempty, unshuffled held-out test loader.
        device: Device used for representation extraction.

    Returns:
        Result records in the original CSV/JSON field order.

    Notes:
        No scaler or classifier is fitted here. Feature matrices are released
        after each condition to limit peak retained memory.
    """
    rows: list[dict[str, Any]] = []

    for selected in selected_probes:
        condition = selected.condition
        test_features, test_labels = extract_features(
            condition, test_loader, device
        )
        test_features = selected.scaler.transform(test_features)
        probabilities = selected.classifier.predict_proba(test_features)[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        metrics = binary_metrics(test_labels, predictions)

        rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "layer": condition.layer,
                "feature_count": selected.feature_count,
                "selected_c": selected.selected_c,
                "validation_accuracy": selected.validation_accuracy,
                "validation_macro_f1": selected.validation_macro_f1,
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_footwear_f1": metrics["footwear_f1"],
                "true_negative": metrics["true_negative"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "true_positive": metrics["true_positive"],
            }
        )
        print(
            f"Test {condition.name:15s}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}"
        )

        del test_features, test_labels, probabilities, predictions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def majority_baseline_row(test_labels: np.ndarray) -> dict[str, Any]:
    """Evaluate the fixed always-non-footwear baseline.

    Args:
        test_labels: Binary targets of the held-out test examples.

    Returns:
        Baseline record using the same fields as learned-probe results.

    Notes:
        The prediction is fixed at zero; the test labels do not select which
        class to predict. Non-footwear is the majority under the experiment's
        original-class-stratified sampling.
    """
    predictions = np.zeros_like(test_labels)
    metrics = binary_metrics(test_labels, predictions)

    return {
        "condition": "always_non_footwear",
        "family": "baseline",
        "layer": "constant",
        "feature_count": 0,
        "selected_c": None,
        "validation_accuracy": None,
        "validation_macro_f1": None,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_footwear_f1": metrics["footwear_f1"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
    }