"""Linear probes with training-only scaling and validation-only selection."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from .extension_config import ExtensionConfig


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, n_classes: int
) -> dict:
    """Return accuracy, macro-F1, balanced accuracy, per-class F1, and counts.

    Args:
        targets: True task labels from 0 to n_classes-1.
        predictions: Predicted task labels in the same coding.
        n_classes: Number of downstream target classes.

    Returns:
        Scalar metrics and a confusion matrix whose rows are true labels.

    Notes:
        Balanced accuracy is the mean recall over the explicit class set.
        This matters for pooled cross-category validation/test distributions.
    """
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            )
        ),
        "balanced_accuracy": float(
            recall_score(
                targets, predictions, labels=labels, average="macro", zero_division=0
            )
        ),
        "per_class_f1": f1_score(
            targets, predictions, labels=labels, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
    }


def fit_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    c_value: float,
    config: ExtensionConfig,
    seed: int,
) -> LogisticRegression:
    """Fit L2 logistic regression, retrying once after a convergence warning.

    Args:
        features: Standardized probe-training matrix.
        targets: Task-specific training labels.
        c_value: Inverse regularization strength.
        config: Initial iteration limit.
        seed: Recorded estimator random-state value.

    Returns:
        A converged classifier.

    Raises:
        RuntimeError: If both iteration budgets fail to converge.
    """
    for max_iter in (config.probe_max_iter, config.probe_max_iter * 5):
        # Omitting penalty uses the L2 default and avoids newer-version
        # deprecation warnings without changing the intended regularization.
        classifier = LogisticRegression(
            C=c_value, solver="lbfgs", max_iter=max_iter, random_state=seed
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(features, targets)
        converged = True
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
            else:
                warnings.warn(str(warning.message), warning.category, stacklevel=2)
        if converged:
            return classifier
    raise RuntimeError(
        f"Probe C={c_value} did not converge. Increase --probe-max-iter."
    )


def fit_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    config: ExtensionConfig,
    seed: int,
    n_classes: int,
) -> dict:
    """Select a probe by validation macro-F1, without changing feature caches.

    Args:
        train_features: Only the small labeled probe-training subset.
        train_targets: Labels for that subset.
        val_features: Fixed validation features, never used to fit scaling.
        val_targets: Fixed validation labels used to choose C.
        config: Candidate C values and solver settings.
        seed: Estimator seed.
        n_classes: Number of task labels.

    Returns:
        Fitted scaler/classifier, selected C, validation metrics, and C search.

    Raises:
        ValueError: If training or validation omits a target class.
        RuntimeError: If a classifier fails to converge.
    """
    expected = set(range(n_classes))
    if set(train_targets.tolist()) != expected or set(val_targets.tolist()) != expected:
        raise ValueError(
            "Probe training and validation must each contain every task class."
        )
    scaler = StandardScaler(copy=True)
    train = scaler.fit_transform(np.asarray(train_features, dtype=np.float64))
    validation = scaler.transform(np.asarray(val_features, dtype=np.float64))
    best = None
    search = []
    for c_value in config.probe_cs:
        classifier = fit_classifier(train, train_targets, c_value, config, seed)
        scores = classification_metrics(
            val_targets, classifier.predict(validation), n_classes
        )
        search.append({"c": c_value, **scores})
        if best is None or scores["macro_f1"] > best["validation"]["macro_f1"]:
            best = {
                "scaler": scaler,
                "classifier": classifier,
                "selected_c": float(c_value),
                "validation": scores,
            }
    if best is None:
        raise ValueError("At least one C candidate is required.")
    best["c_search"] = search
    return best
