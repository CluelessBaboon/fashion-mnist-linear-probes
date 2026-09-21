"""Focused tests for the add-on low-data probe experiment."""

from __future__ import annotations

import unittest

import numpy as np

from fashion_probes.extension_config import make_tasks
from fashion_probes.extension_data import task_targets
from fashion_probes.extension_sample_efficiency import (
    CONDITION_ORDER,
    DEFAULT_SAMPLE_SIZES,
    aggregate_paired_differences,
    aggregate_results,
    nested_task_samples,
    paired_differences,
    parse_arguments,
)


class ExtensionSampleEfficiencyTests(unittest.TestCase):
    """Protect exact budgets, pairing, and reproducible nested sampling."""

    def test_default_design_uses_exact_low_total_budgets(self) -> None:
        """Defaults match the requested 10--200 total-example design."""
        config = parse_arguments([])
        self.assertEqual(config.sample_sizes, DEFAULT_SAMPLE_SIZES)
        self.assertEqual(config.controlled_dim, 32)
        self.assertEqual(config.projection_seeds, (0, 1, 2, 3, 4))
        self.assertEqual(config.repeats, 10)
        self.assertEqual(config.source_seed, 42)

    def test_nested_samples_are_exact_near_balanced_and_reproducible(self) -> None:
        """Every task uses exact totals, nested prefixes, and class balance."""
        labels = np.repeat(np.arange(10), 100)
        tasks = make_tasks()
        sizes = (10, 20, 30, 50, 80, 110, 150, 200)
        pool, samples = nested_task_samples(
            np.arange(len(labels)), labels, tasks, sizes, 3, 42
        )
        other_pool, other_samples = nested_task_samples(
            np.arange(len(labels)), labels, tasks, sizes, 3, 42
        )
        np.testing.assert_array_equal(pool, other_pool)
        for repeat in range(3):
            for task in tasks:
                previous: set[int] = set()
                for size in sizes:
                    positions = samples[(repeat, task.name, size)]
                    ids = pool[positions]
                    self.assertEqual(len(ids), size)
                    self.assertEqual(len(set(ids.tolist())), size)
                    self.assertTrue(previous <= set(ids.tolist()))
                    mask, targets = task_targets(labels[ids], task)
                    self.assertTrue(mask.all())
                    counts = np.bincount(targets, minlength=len(task.groups))
                    self.assertLessEqual(int(counts.max() - counts.min()), 1)
                    np.testing.assert_array_equal(
                        positions, other_samples[(repeat, task.name, size)]
                    )
                    if task.category == "cross_pooled":
                        pooled_ids = ids[targets == 0]
                        original_counts = [
                            int(np.sum(labels[pooled_ids] == label))
                            for label in task.groups[0]
                        ]
                        self.assertLessEqual(max(original_counts) - min(original_counts), 1)
                    previous = set(ids.tolist())

    def test_aggregate_and_matched_differences(self) -> None:
        """Aggregation counts all 10x5 runs and preserves matched differences."""
        rows = []
        for repeat in range(1, 11):
            for projection_seed in range(5):
                for family, score in (("random", 0.70), ("fine", 0.82), ("coarse", 0.88)):
                    rows.append(
                        {
                            "source_seed": 42,
                            "repeat": repeat,
                            "sampling_seed": 41 + repeat,
                            "projection_seed": projection_seed,
                            "task": "sandal_vs_sneaker",
                            "task_category": "within_footwear",
                            "sample_size": 20,
                            "condition": f"{family}_fc",
                            "family": family,
                            "layer": "fc",
                            "original_feature_count": 128,
                            "feature_count": 32,
                            "selected_c": 0.1,
                            "test_accuracy": score,
                            "test_macro_f1": score,
                            "test_balanced_accuracy": score,
                        }
                    )
        aggregate = aggregate_results(rows)
        self.assertEqual(len(aggregate), 3)
        self.assertTrue(all(row["n_runs"] == 50 for row in aggregate))
        self.assertTrue(all(row["n_sampling_repeats"] == 10 for row in aggregate))
        self.assertTrue(all(row["n_projection_seeds"] == 5 for row in aggregate))
        paired = paired_differences(rows)
        self.assertEqual(len(paired), 100)
        summary = aggregate_paired_differences(paired)
        self.assertEqual(len(summary), 2)
        by_family = {row["family"]: row for row in summary}
        self.assertAlmostEqual(by_family["fine"]["test_macro_f1_mean"], 0.12)
        self.assertAlmostEqual(by_family["coarse"]["test_macro_f1_mean"], 0.18)
        self.assertEqual(CONDITION_ORDER[-1], "coarse_fc")


if __name__ == "__main__":
    unittest.main()
