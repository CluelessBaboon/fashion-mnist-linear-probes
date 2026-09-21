"""Serialization and diagnostic plots using the original output formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .config import LAYER_NAMES
from .probes import SelectedProbe


FIGURE_DPI = 220

# A restrained, high-contrast palette chosen specifically for this project.
# It avoids Matplotlib's default blue/orange cycle while remaining legible in
# grayscale through distinct markers and line styles.
PAPER = "#FBFAF7"
INK = "#20242B"
MUTED_INK = "#656B73"
GRID = "#D8D5CE"
RANDOM_COLOR = "#665191"  # aubergine
TRAINED_COLOR = "#D65F45"  # terracotta
PIXEL_COLOR = "#087E8B"  # petrol
BASELINE_COLOR = "#78716C"  # warm slate
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "ivory_to_aubergine",
    ("#F4EFE6", "#C7B8D9", "#665191", "#2E2340"),
)


def _style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    """Apply the shared minimal figure style without global side effects."""
    axis.set_facecolor(PAPER)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED_INK)
    axis.spines["bottom"].set_color(MUTED_INK)
    axis.tick_params(colors=INK, labelsize=9)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, png_path: Path) -> None:
    """Save a high-resolution PNG and a vector PDF with matching content."""
    figure.savefig(
        png_path,
        dpi=FIGURE_DPI,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    figure.savefig(
        png_path.with_suffix(".pdf"),
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )


def json_default(value: Any) -> Any:
    """Convert supported nonstandard values into JSON-compatible objects.

    Args:
        value: Path, NumPy array, or NumPy scalar requiring conversion.

    Returns:
        Corresponding string, list, or Python scalar.

    Raises:
        TypeError: If the value's type is unsupported.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def write_json(path: Path, content: Any) -> None:
    """Write indented JSON using the original serialization rules.

    Args:
        path: Destination file; its parent directory must exist.
        content: JSON-compatible data, optionally containing supported
            Path or NumPy values.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If content contains an unsupported value.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, default=json_default)


def save_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write result records with the first record's field order.

    Args:
        rows: Nonempty records with matching keys.
        path: Destination CSV file.

    Raises:
        OSError: If the file cannot be written.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_curves(
    history: list[dict[str, float | int]],
    output_dir: Path,
) -> None:
    """Save CNN training and validation loss/accuracy curves.

    Args:
        history: Per-epoch records returned by train_cnn.
        output_dir: Existing destination directory.

    Raises:
        OSError: If the figure cannot be saved.
    """
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=PAPER)

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        color=RANDOM_COLOR,
        marker="o",
        markersize=4,
        linewidth=2.1,
        label="Train",
    )
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        color=TRAINED_COLOR,
        marker="D",
        markersize=3.7,
        linewidth=2.1,
        label="Validation",
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        color=RANDOM_COLOR,
        marker="o",
        markersize=4,
        linewidth=2.1,
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] for row in history],
        color=TRAINED_COLOR,
        marker="D",
        markersize=3.7,
        linewidth=2.1,
        label="Validation",
    )
    axes[1].set(
        title="Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.75, 1.0),
    )
    for axis in axes:
        _style_axis(axis)
        axis.set_xticks(epochs)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle("CNN training diagnostics", color=INK, y=1.08, fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output_dir / "cnn_learning_curves.png")
    plt.close(figure)


def save_probe_comparison(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot random and trained layer probes against pixels and the baseline.

    Args:
        rows: Records for all seven feature conditions and the fixed baseline.
        output_dir: Existing destination directory.

    Raises:
        KeyError: If an expected condition is missing.
        OSError: If the figure cannot be saved.
    """
    by_condition = {row["condition"]: row for row in rows}
    ordered_conditions = (
        "always_non_footwear",
        "pixels",
        "random_conv1",
        "trained_conv1",
        "random_conv2",
        "trained_conv2",
        "random_fc",
        "trained_fc",
    )
    display_names = {
        "always_non_footwear": "Constant baseline",
        "pixels": "Raw pixels",
        "random_conv1": "Conv1 · random",
        "trained_conv1": "Conv1 · trained",
        "random_conv2": "Conv2 · random",
        "trained_conv2": "Conv2 · trained",
        "random_fc": "FC · random",
        "trained_fc": "FC · trained",
    }
    colors = {
        "baseline": BASELINE_COLOR,
        "pixels": PIXEL_COLOR,
        "random": RANDOM_COLOR,
        "trained": TRAINED_COLOR,
    }
    markers = {"baseline": "s", "pixels": "P", "random": "o", "trained": "D"}

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5.4),
        facecolor=PAPER,
        gridspec_kw={"width_ratios": (1.05, 1.35)},
    )

    # Full-range panel preserves baseline context without exaggerating the
    # near-ceiling probe scores.
    y_positions = np.arange(len(ordered_conditions))
    for y_position, condition in zip(y_positions, ordered_conditions):
        row = by_condition[condition]
        score = float(row["test_macro_f1"])
        family = row["family"]
        axes[0].scatter(
            score,
            y_position,
            s=64,
            color=colors[family],
            marker=markers[family],
            edgecolor=PAPER,
            linewidth=0.8,
            zorder=3,
        )
        axes[0].text(
            1.018,
            y_position,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
        )
    axes[0].set_yticks(
        y_positions,
        [display_names[name] for name in ordered_conditions],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.38, 1.075)
    axes[0].set_xlabel("Held-out test macro-F1")
    axes[0].set_title("All conditions", color=INK, loc="left")
    _style_axis(axes[0], grid_axis="x")

    # The second panel explicitly marks its magnification and prints exact
    # values so very small matched differences remain interpretable.
    layers = tuple(LAYER_NAMES)
    layer_labels = ("Conv1 · 3,136D", "Conv2 · 1,568D", "FC · 128D")
    y_layers = np.arange(len(layers))
    random_scores = np.asarray(
        [by_condition[f"random_{layer}"]["test_macro_f1"] for layer in layers]
    )
    trained_scores = np.asarray(
        [by_condition[f"trained_{layer}"]["test_macro_f1"] for layer in layers]
    )
    all_cnn_scores = np.concatenate((random_scores, trained_scores))
    padding = max(0.00035, float(np.ptp(all_cnn_scores)) * 0.22)
    zoom_min = max(0.0, float(all_cnn_scores.min()) - padding)
    zoom_max = min(1.0002, float(all_cnn_scores.max()) + padding)

    for y_position, random_score, trained_score in zip(
        y_layers, random_scores, trained_scores
    ):
        axes[1].plot(
            (random_score, trained_score),
            (y_position, y_position),
            color=GRID,
            linewidth=2.2,
            zorder=1,
        )
        axes[1].scatter(
            random_score,
            y_position,
            s=76,
            color=RANDOM_COLOR,
            marker="o",
            edgecolor=PAPER,
            linewidth=0.9,
            zorder=3,
        )
        axes[1].scatter(
            trained_score,
            y_position,
            s=70,
            color=TRAINED_COLOR,
            marker="D",
            edgecolor=PAPER,
            linewidth=0.9,
            zorder=3,
        )
        axes[1].annotate(
            f"{random_score:.4f}",
            (random_score, y_position),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=RANDOM_COLOR,
        )
        axes[1].annotate(
            f"{trained_score:.4f}",
            (trained_score, y_position),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=TRAINED_COLOR,
        )
    pixel_score = float(by_condition["pixels"]["test_macro_f1"])
    if zoom_min <= pixel_score <= zoom_max:
        axes[1].axvline(
            pixel_score,
            color=PIXEL_COLOR,
            linewidth=1.4,
            linestyle=(0, (3, 2)),
        )
    axes[1].set_yticks(y_layers, layer_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(zoom_min, zoom_max)
    axes[1].set_xlabel("Held-out test macro-F1 (zoomed scale)")
    axes[1].set_title("Matched CNN layers", color=INK, loc="left")
    _style_axis(axes[1], grid_axis="x")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=RANDOM_COLOR,
            markeredgecolor=PAPER,
            markersize=8,
            label="Random CNN",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=TRAINED_COLOR,
            markeredgecolor=PAPER,
            markersize=7,
            label="Trained CNN",
        ),
        Line2D(
            [0],
            [0],
            color=PIXEL_COLOR,
            linewidth=1.5,
            linestyle=(0, (3, 2)),
            label="Raw pixels",
        ),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.16),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Linear accessibility of the footwear category",
        color=INK,
        fontsize=15,
        y=1.01,
    )
    figure.text(
        0.5,
        -0.015,
        "Frozen representations · logistic probes · official 10,000-image test set",
        ha="center",
        color=MUTED_INK,
        fontsize=9,
    )
    figure.tight_layout(w_pad=3.0)
    _save_figure(figure, output_dir / "probe_macro_f1_comparison.png")
    plt.close(figure)


def save_confusion_matrices(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save raw binary confusion counts in the original eight-panel layout.

    Args:
        rows: Up to eight evaluation records containing confusion counts.
        output_dir: Existing destination directory.

    Raises:
        OSError: If the figure cannot be saved.
    """
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), facecolor=PAPER)

    for axis, row in zip(axes.ravel(), rows):
        matrix = np.asarray(
            [
                [row["true_negative"], row["false_positive"]],
                [row["false_negative"], row["true_positive"]],
            ]
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        proportions = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        axis.imshow(proportions, cmap=CONFUSION_CMAP, vmin=0, vmax=1)

        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    (
                        f"{matrix[row_index, column_index]:,}\n"
                        f"{proportions[row_index, column_index]:.1%}"
                    ),
                    ha="center",
                    va="center",
                    color=(
                        PAPER
                        if proportions[row_index, column_index] >= 0.55
                        else INK
                    ),
                    fontsize=9,
                )

        condition_label = (
            row["condition"]
            .replace("_", " ")
            .title()
            .replace("Fc", "FC")
        )
        axis.set_title(
            condition_label,
            color=INK,
            fontsize=10,
        )
        axis.set_xticks(
            (0, 1), ("Non-footwear", "Footwear"), rotation=18
        )
        axis.set_yticks((0, 1), ("Non-footwear", "Footwear"))
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.tick_params(colors=INK, labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)

    for axis in axes.ravel()[len(rows):]:
        axis.axis("off")

    figure.suptitle(
        "Probe confusion matrices",
        color=INK,
        fontsize=15,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.015,
        "Cells show count and percentage within each actual class",
        ha="center",
        color=MUTED_INK,
        fontsize=9,
    )
    figure.tight_layout()
    _save_figure(figure, output_dir / "probe_confusion_matrices.png")
    plt.close(figure)


def save_probe_objects(
    selected_probes: list[SelectedProbe],
    output_dir: Path,
) -> None:
    """Serialize selected probes using the original payload fields.

    Args:
        selected_probes: Fitted probes and their associated metadata.
        output_dir: Parent directory for the probes subdirectory.

    Raises:
        OSError: If a directory or serialized probe cannot be written.

    Notes:
        Payloads contain scikit-learn objects and plain metadata, rather than
        instances of the project's dataclasses.
    """
    probe_dir = output_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    for selected in selected_probes:
        payload = {
            "condition": selected.condition.name,
            "layer": selected.condition.layer,
            "feature_count": selected.feature_count,
            "selected_c": selected.selected_c,
            "validation_accuracy": selected.validation_accuracy,
            "validation_macro_f1": selected.validation_macro_f1,
            "scaler": selected.scaler,
            "classifier": selected.classifier,
        }
        joblib.dump(
            payload,
            probe_dir / f"{selected.condition.name}.joblib",
        )
