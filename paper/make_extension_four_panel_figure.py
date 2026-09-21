"""Create a four-panel comparison of the two coarse-training objectives."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, ScalarFormatter


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "figures" / "extension_objective_four_panel"
EXPERIMENTS = (
    (
        "Footwear subclasses (3-way)",
        "Footwear vs. non-footwear",
        PROJECT
        / "results"
        / "extension_probes_seed42_n10_200_dim32_complete_20260918"
        / "extension_probe_raw.csv",
        "footwear_three_way",
    ),
    (
        "Tops subclasses (4-way)",
        "Tops vs. non-tops",
        PROJECT
        / "results"
        / "extension_tops_probes_seed42_n10_200_dim32_20260921"
        / "extension_probe_raw.csv",
        "tops_four_way",
    ),
)
FAMILIES = ("pixels", "random", "fine", "coarse")

PAPER = "#FBFAF7"
INK = "#20242B"
MUTED_INK = "#656B73"
GRID = "#D8D5CE"
STYLES = {
    "pixels": ("Pixels", "#087E8B", "P", (0, (3, 2))),
    "random": ("Random CNN", "#665191", "o", (0, (1, 2))),
    "fine": ("Ten-class CNN", "#D65F45", "D", "-"),
    "coarse": ("Matched coarse CNN", "#B07A2A", "^", "-"),
}


def load_curves(path: Path, within_task: str) -> dict[tuple[str, int, int, int, str], float]:
    """Load within-superclass and mean boundary scores for each matched run.

    Args:
        path: Raw extension-results CSV.
        within_task: Identifier of the superclass multiclass task.

    Returns:
        Scores keyed by panel, repeat, projection seed, label budget, and family.

    Raises:
        ValueError: If any run lacks its within-superclass or boundary scores.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["family"] in FAMILIES
            and (row["layer"] == "fc" or row["family"] == "pixels")
        ]

    grouped: dict[tuple[int, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["repeat"]),
                int(row["projection_seed"]),
                int(row["sample_size"]),
                row["family"],
            )
        ].append(row)

    values = {}
    for (repeat, projection, size, family), group in grouped.items():
        within = [float(row["test_macro_f1"]) for row in group if row["task"] == within_task]
        boundary = [
            float(row["test_macro_f1"])
            for row in group
            if row["task_category"] == "cross_pooled"
        ]
        if len(within) != 1 or not boundary:
            raise ValueError(f"Incomplete {family} results in {path}.")
        values[("within", repeat, projection, size, family)] = within[0]
        values[("boundary", repeat, projection, size, family)] = float(np.mean(boundary))
    return values


def summarize(
    values: dict[tuple[str, int, int, int, str], float], panel: str, family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return label budgets and the mean and sample SD across matched runs."""
    sizes = np.asarray(
        sorted(
            {
                size
                for task_type, _, _, size, row_family in values
                if task_type == panel and row_family == family
            }
        ),
        dtype=int,
    )
    runs = np.asarray(
        [
            [
                value
                for (task_type, _, _, size, row_family), value in values.items()
                if task_type == panel and row_family == family and size == n
            ]
            for n in sizes
        ],
        dtype=float,
    )
    return sizes, runs.mean(axis=1), runs.std(axis=1, ddof=1)


def style_axis(axis: plt.Axes) -> None:
    """Apply the shared publication style to one subplot."""
    axis.set_facecolor(PAPER)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED_INK)
    axis.spines["bottom"].set_color(MUTED_INK)
    axis.tick_params(colors=INK, labelsize=6.5)
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def main() -> None:
    """Create PDF and PNG versions of the four-panel paper figure."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 3.65),
        sharex=True,
        sharey=True,
        facecolor=PAPER,
        constrained_layout=True,
    )

    for row, (within_title, boundary_title, path, within_task) in enumerate(EXPERIMENTS):
        values = load_curves(path, within_task)
        for column, (panel, title) in enumerate(
            (("within", within_title), ("boundary", boundary_title))
        ):
            axis = axes[row, column]
            for family in FAMILIES:
                sizes, mean, sd = summarize(values, panel, family)
                label, color, marker, linestyle = STYLES[family]
                axis.plot(
                    sizes,
                    mean,
                    color=color,
                    marker=marker,
                    markersize=3.2,
                    linewidth=1.35,
                    linestyle=linestyle,
                    label=label,
                )
                axis.fill_between(
                    sizes,
                    np.maximum(mean - sd, 0),
                    np.minimum(mean + sd, 1),
                    color=color,
                    alpha=0.09,
                    linewidth=0,
                )
            axis.set_title(title, fontsize=8.2)
            axis.set_xscale("log")
            axis.set_ylim(0.4, 1.005)
            axis.set_xticks((10, 20, 50, 110, 200))
            axis.get_xaxis().set_major_formatter(ScalarFormatter())
            axis.get_xaxis().set_minor_locator(NullLocator())
            style_axis(axis)

    for axis in axes[:, 0]:
        axis.set_ylabel("Test macro-F1", fontsize=7.2)
    for axis in axes[1, :]:
        axis.set_xlabel("Probe-training examples (total)", fontsize=7.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=4,
        frameon=False,
        fontsize=7.0,
    )
    figure.suptitle(
        "Training objective controls downstream accessibility",
        color=INK,
        fontsize=10.2,
        y=1.09,
    )
    figure.text(
        0.5,
        -0.025,
        "Boundary panels average the constituent class-vs.-remainder tasks; bands show ±1 sample SD over 50 matched runs",
        ha="center",
        color=MUTED_INK,
        fontsize=6.3,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    figure.savefig(
        OUTPUT.with_suffix(".png"), dpi=300, facecolor=PAPER, bbox_inches="tight"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
