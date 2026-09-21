"""Check extension controls and run a small synthetic end-to-end pipeline.

Run from the project root with:
    python -m unittest discover -s tests -p "test_extension.py" -v

The synthetic integration test verifies execution, not scientific performance.
No download or original-script backup is needed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch
from fashion_probes import extension_pipeline as pipeline
from fashion_probes.extension_config import make_tasks, parse_arguments
from fashion_probes.extension_data import make_probe_samples, make_splits, task_targets
from fashion_probes.extension_probes import fit_probe
from fashion_probes.extension_training import cnn_targets, freeze, make_models
from torch.utils.data import Dataset


class SyntheticImages(Dataset):
    """Deterministic tiny image fixture carrying all ten original labels."""

    def __init__(self, per_class: int, seed: int) -> None:
        """Create noisy image patterns with a labels/targets interface."""
        self.targets = torch.arange(10).repeat_interleave(per_class)
        generator = torch.Generator().manual_seed(seed)
        self.images = (
            torch.rand(len(self.targets), 1, 28, 28, generator=generator) * 0.1
        )
        for index, label in enumerate(self.targets.tolist()):
            self.images[index, :, label : label + 2, :] += 0.6

    def __len__(self) -> int:
        """Return the number of fixture examples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an image and its original class label."""
        return self.images[index], self.targets[index]


class ExtensionTests(unittest.TestCase):
    """Verify scientific controls that a refactor or extension can break."""

    def test_task_definitions_and_filtering(self) -> None:
        """Unrelated footwear must be excluded from each cross-category task."""
        tasks = make_tasks()
        self.assertEqual(len(tasks), 7)
        self.assertEqual(len(make_tasks(True)), 28)
        original = np.arange(10)
        mask, labels = task_targets(original, tasks[0])
        np.testing.assert_array_equal(original[mask], [5, 7, 9])
        np.testing.assert_array_equal(labels, [0, 1, 2])
        sandal = next(task for task in tasks if task.name == "sandal_vs_non_footwear")
        mask, labels = task_targets(original, sandal)
        self.assertFalse(mask[7] or mask[9])
        self.assertEqual(int(labels.sum()), 1)
        np.testing.assert_array_equal(
            cnn_targets(torch.arange(10), "coarse").numpy(),
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        )

        tops = make_tasks(coarse_target="tops")
        self.assertEqual(len(tops), 11)
        self.assertEqual(len(make_tasks(True, "tops")), 35)
        mask, labels = task_targets(original, tops[0])
        np.testing.assert_array_equal(original[mask], [0, 2, 4, 6])
        np.testing.assert_array_equal(labels, [0, 1, 2, 3])
        tshirt = next(task for task in tops if task.name == "tshirt_vs_non_tops")
        mask, labels = task_targets(original, tshirt)
        self.assertFalse(mask[2] or mask[4] or mask[6])
        self.assertEqual(int(labels.sum()), 1)
        np.testing.assert_array_equal(
            cnn_targets(torch.arange(10), "coarse", "tops").numpy(),
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        )

    def test_nested_balanced_shared_samples(self) -> None:
        """Budgets are balanced/nested and reproducible across representations."""
        labels = np.repeat(np.arange(10), 25)
        tasks = make_tasks(True)
        budgets = (2, 5, 10)
        pool, samples = make_probe_samples(
            np.arange(len(labels)), labels, tasks, budgets, 42
        )
        other_pool, other_samples = make_probe_samples(
            np.arange(len(labels)), labels, tasks, budgets, 42
        )
        np.testing.assert_array_equal(pool, other_pool)
        for task in tasks:
            previous = set()
            for budget in budgets:
                positions = samples[(task.name, budget)]
                ids = pool[positions]
                self.assertTrue(previous <= set(ids))
                self.assertEqual(len(set(ids)), len(ids))
                mask, targets = task_targets(labels[ids], task)
                self.assertTrue(mask.all())
                np.testing.assert_array_equal(
                    np.bincount(targets), np.full(len(task.groups), budget)
                )
                np.testing.assert_array_equal(
                    positions, other_samples[(task.name, budget)]
                )
                if task.category == "cross_pooled":
                    counts = [
                        int(np.sum(labels[ids] == label)) for label in task.groups[0]
                    ]
                    self.assertLessEqual(max(counts) - min(counts), 1)
                previous = set(ids)

    def test_split_isolation_and_multiple_pilot_exclusions(self) -> None:
        """No training/validation overlap or previously inspected test IDs."""
        labels = np.tile(np.arange(10), 20)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.npz"
            second = Path(temporary) / "second.npz"
            np.savez(first, test=np.arange(10))
            np.savez(second, test=np.arange(5, 15))
            config = replace(
                parse_arguments(["--quick"]),
                cnn_train_size=100,
                cnn_val_size=50,
                probe_val_size=30,
                test_size=30,
                exclude_test_indices=(first, second),
            )
            splits = make_splits(labels, labels, config)
            self.assertFalse(set(splits["cnn_train"]) & set(splits["cnn_validation"]))
            self.assertTrue(
                set(splits["probe_validation"]) <= set(splits["cnn_validation"])
            )
            self.assertFalse(set(splits["test"]) & set(range(15)))
            np.testing.assert_array_equal(splits["excluded_test"], np.arange(15))

    def test_identical_independent_backbones(self) -> None:
        """The copied binary head must not alter initialization of probed layers."""
        models = make_models(42)
        for layer in ("conv1", "conv2", "fc"):
            reference = getattr(models["random"], layer)
            for family in ("fine", "coarse"):
                candidate = getattr(models[family], layer)
                for expected, actual in zip(
                    reference.parameters(), candidate.parameters()
                ):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    self.assertNotEqual(expected.data_ptr(), actual.data_ptr())
        inputs = torch.zeros(2, 1, 28, 28)
        self.assertEqual(tuple(models["fine"](inputs)[0].shape), (2, 10))
        self.assertEqual(tuple(models["coarse"](inputs)[0].shape), (2, 2))
        freeze(models["coarse"])
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in models["coarse"].parameters()
            )
        )

    def test_scaler_isolation_and_cache_integrity(self) -> None:
        """Scaler statistics must use the small training sample alone."""
        config = parse_arguments(["--quick"])
        training = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        validation = training + 100.0
        labels = np.array([0, 0, 1, 1])
        before_train, before_val = training.copy(), validation.copy()
        selected = fit_probe(training, labels, validation, labels, config, 42, 2)
        np.testing.assert_allclose(selected["scaler"].mean_, before_train.mean(axis=0))
        np.testing.assert_array_equal(training, before_train)
        np.testing.assert_array_equal(validation, before_val)

    def test_every_seed_is_fitted_before_test_evaluation(self) -> None:
        """Execute the actual pipeline with two seeds and synthetic datasets."""
        events = []
        original_fit, original_evaluate = pipeline.fit_seed, pipeline.evaluate_seed

        def tracked_fit(seed, *args, **kwargs):
            """Record fitting order while executing the real fitting function."""
            events.append(("fit", seed))
            return original_fit(seed, *args, **kwargs)

        def tracked_evaluate(seed, *args, **kwargs):
            """Record evaluation order while executing real saved-probe loading."""
            events.append(("test", seed))
            return original_evaluate(seed, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            config = parse_arguments(
                [
                    "--quick",
                    "--seeds",
                    "11",
                    "12",
                    "--device",
                    "cpu",
                    "--cnn-train-size",
                    "100",
                    "--cnn-val-size",
                    "50",
                    "--probe-val-size",
                    "30",
                    "--test-size",
                    "30",
                    "--probe-sizes",
                    "2",
                    "--output-dir",
                    str(output),
                ]
            )
            with (
                patch.object(pipeline, "fit_seed", side_effect=tracked_fit),
                patch.object(pipeline, "evaluate_seed", side_effect=tracked_evaluate),
                patch(
                    "fashion_probes.extension_reporting.save_figure",
                    side_effect=lambda figure, stem: plt.close(figure),
                ),
            ):
                pipeline.run(config, SyntheticImages(20, 1), SyntheticImages(10, 2))
            self.assertEqual(
                events, [("fit", 11), ("fit", 12), ("test", 11), ("test", 12)]
            )
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "complete")
            self.assertEqual(status["dataset_class"], "SyntheticImages")
            rows = json.loads((output / "probe_results.json").read_text())
            self.assertEqual(len(rows), 2 * 7 * 11)
            self.assertTrue((output / "seed_11" / "coarse_cnn.pt").is_file())
            with self.assertRaises(FileExistsError):
                pipeline.run(config, SyntheticImages(20, 1), SyntheticImages(10, 2))


if __name__ == "__main__":
    unittest.main()
