"""Auditable result tables and publication-friendly extension figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .extension_config import LAYERS, Task


def json_default(value: Any) -> Any:
    """Convert paths, dataclasses, and NumPy values; reject other objects."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def save_json(path: Path, value: Any) -> None:
    """Write JSON with finite numerical values; parent directory must exist."""
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict]) -> None:
    """Write records with a union of fields; encode nested values as JSON."""
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, default=json_default)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    """Save a figure as 300-dpi PNG and vector PDF, then close it."""
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history: list[dict], stem: Path) -> None:
    """Plot training and validation loss and accuracy for one CNN."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for prefix, label in (("train", "Training"), ("validation", "Validation")):
            axis.plot(
                [row["epoch"] for row in history],
                [row[f"{prefix}_{metric}"] for row in history],
                marker="o",
                label=label,
            )
        axis.set(
            xlabel="Epoch", ylabel="Cross-entropy" if metric == "loss" else "Accuracy"
        )
        axis.legend()
    axes[1].set_ylim(0, 1)
    save_figure(figure, stem)


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate metrics across seeds, reporting sample SD rather than a CI.

    Args:
        rows: Per-seed test or paired-difference records.

    Returns:
        One record for each task/condition/budget combination.
    """
    fields = (
        "task",
        "task_category",
        "family",
        "layer",
        "examples_per_class",
        "n_probe_train",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        row = dict(zip(fields, key))
        row["n_seeds"] = len(group)
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = np.asarray([item[f"test_{metric}"] for item in group])
            row[f"test_{metric}_mean"] = float(values.mean())
            row[f"test_{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        result.append(row)
    return result


def paired_differences(rows: list[dict]) -> list[dict]:
    """Compare trained and random representations at the same seed/layer/budget."""
    keys = ("seed", "task", "layer", "examples_per_class")
    random = {
        tuple(row[key] for key in keys): row
        for row in rows
        if row["family"] == "random"
    }
    result = []
    for row in rows:
        if row["family"] not in ("fine", "coarse"):
            continue
        reference = random[tuple(row[key] for key in keys)]
        difference = {
            key: row[key] for key in (*keys, "task_category", "family", "n_probe_train")
        }
        difference["comparison"] = f"{row['family']}_minus_random"
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            difference[f"test_{metric}"] = (
                row[f"test_{metric}"] - reference[f"test_{metric}"]
            )
        result.append(difference)
    return result


def plot_probe_curves(
    summary: list[dict], task: Task, directory: Path, metric: str = "macro_f1"
) -> None:
    """Plot means with between-seed SD at each probe budget and hidden layer."""
    rows = [row for row in summary if row["task"] == task.name]
    figure, axes = plt.subplots(
        1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True
    )
    displays = (
        ("random", "Random CNN", "tab:gray"),
        ("fine", "Ten-class CNN", "tab:blue"),
        ("coarse", "Coarse CNN", "tab:orange"),
        ("pixels", "Pixels", "black"),
        ("constant", "Constant baseline", "tab:green"),
    )
    for axis, layer in zip(axes, LAYERS):
        for family, label, color in displays:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["family"] == family
                    and (family in ("pixels", "constant") or row["layer"] == layer)
                ],
                key=lambda row: row["examples_per_class"],
            )
            if not selected:
                continue
            x = [row["examples_per_class"] for row in selected]
            mean = np.asarray([row[f"test_{metric}_mean"] for row in selected])
            sd = np.asarray([row[f"test_{metric}_std"] for row in selected])
            axis.plot(
                x,
                mean,
                marker="o",
                label=label,
                color=color,
                linestyle="--" if family in ("pixels", "constant") else "-",
            )
            if selected[0]["n_seeds"] > 1:
                axis.fill_between(
                    x,
                    np.maximum(mean - sd, 0),
                    np.minimum(mean + sd, 1),
                    color=color,
                    alpha=0.12,
                )
        axis.set(
            title=layer.upper(),
            xlabel="Probe examples per target class",
            xscale="log",
            ylim=(0, 1.02),
        )
        axis.set_xticks(sorted({row["examples_per_class"] for row in rows}))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Test " + metric.replace("_", " "))
    axes[-1].legend(fontsize=7, loc="lower right")
    figure.suptitle(task.name.replace("_", " "))
    save_figure(figure, directory / f"{task.name}_{metric}")


def plot_confusions(rows: list[dict], tasks: list[Task], directory: Path) -> None:
    """Show pixels and FC confusions at the largest budget for this seed.

    Counts and within-true-class proportions are shown together. This is a
    prespecified diagnostic selection, never the best-performing seed/layer.
    """
    for task in tasks:
        candidates = [
            row
            for row in rows
            if row["task"] == task.name
            and (row["family"] == "pixels" or row["layer"] == "fc")
        ]
        largest = max(row["examples_per_class"] for row in candidates)
        selected = [row for row in candidates if row["examples_per_class"] == largest]
        figure, axes = plt.subplots(
            1,
            len(selected),
            figsize=(4 * len(selected), 4),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, row in zip(axes.ravel(), selected):
            counts = np.asarray(row["test_confusion_matrix"])
            proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
            axis.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
            for true in range(len(counts)):
                for predicted in range(len(counts)):
                    value = proportions[true, predicted]
                    axis.text(
                        predicted,
                        true,
                        f"{counts[true, predicted]}\n{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.5 else "black",
                    )
            axis.set_xticks(
                range(len(counts)), task.class_names, rotation=35, ha="right"
            )
            axis.set_yticks(range(len(counts)), task.class_names)
            family_title = {
                "pixels": "Pixels",
                "random": "Random CNN",
                "fine": "Ten-class CNN",
                "coarse": "Coarse CNN",
            }[row["family"]]
            axis.set(title=family_title, xlabel="Predicted", ylabel="True")
        figure.suptitle(
            f"{task.name.replace('_', ' ')}: {largest} training examples per target class"
        )
        save_figure(figure, directory / f"{task.name}_confusion")
