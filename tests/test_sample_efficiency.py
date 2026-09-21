"""Tests for the frozen-checkpoint sample-efficiency experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from fashion_probes import sample_efficiency as experiment


class TinyImages(Dataset):
    """Minimal dataset fixture with a torchvision-compatible targets field."""

    def __init__(self, labels: np.ndarray) -> None:
        self.targets = torch.as_tensor(labels, dtype=torch.int64)
        self.images = torch.zeros(len(labels), 1, 28, 28)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


class SampleEfficiencyTests(unittest.TestCase):
    """Check the sampling, aggregation, and test-isolation controls."""

    def test_prespecified_defaults(self) -> None:
        config = experiment.parse_arguments([])
        self.assertEqual(
            config.sample_sizes,
            (10, 20, 30, 50, 80, 110, 150, 200),
        )
        self.assertEqual(config.repeats, 10)
        self.assertEqual(config.source_dir, Path("results/main_full_seed42"))
        self.assertEqual(experiment.TARGET_NAMES, ("footwear", "tops"))
        self.assertIsNone(config.controlled_dim)
        self.assertEqual(config.projection_seeds, ())

        controlled = experiment.parse_arguments(["--controlled-dim", "32"])
        self.assertEqual(controlled.controlled_dim, 32)
        self.assertEqual(
            controlled.projection_seeds, experiment.DEFAULT_PROJECTION_SEEDS
        )

    def test_random_projection_is_fixed_and_dimension_controlled(self) -> None:
        features = np.arange(40, dtype=np.float32).reshape(5, 8)
        first = experiment.random_project(features, 3, projection_seed=7)
        duplicate = experiment.random_project(features, 3, projection_seed=7)
        alternative = experiment.random_project(features, 3, projection_seed=8)
        self.assertEqual(first.shape, (5, 3))
        np.testing.assert_array_equal(first, duplicate)
        self.assertFalse(np.array_equal(first, alternative))

    def test_tops_mapping_uses_the_four_clothing_top_classes(self) -> None:
        labels = torch.arange(10)
        expected = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
        torch.testing.assert_close(experiment.labels_to_tops(labels), expected)

    def test_repeated_samples_are_ten_class_stratified_and_nested(self) -> None:
        labels = np.repeat(np.arange(10), 50)
        available = np.arange(len(labels))
        sizes = (10, 20, 40, 80)
        samples = experiment.nested_stratified_samples(
            available, labels, sizes, repeats=3, seed=42
        )
        duplicate = experiment.nested_stratified_samples(
            available, labels, sizes, repeats=3, seed=42
        )

        for repeat in range(3):
            previous: set[int] = set()
            for size in sizes:
                selected = samples[(repeat, size)]
                self.assertEqual(len(selected), size)
                self.assertEqual(len(np.unique(selected)), size)
                self.assertTrue(previous <= set(selected.tolist()))
                np.testing.assert_array_equal(
                    np.bincount(labels[selected], minlength=10),
                    np.full(10, size // 10),
                )
                np.testing.assert_array_equal(selected, duplicate[(repeat, size)])
                previous = set(selected.tolist())

    def test_aggregate_uses_between_repeat_sample_sd(self) -> None:
        rows = []
        settings = (
            (1, 0, 0.6, 0.1),
            (1, 1, 0.7, 0.1),
            (2, 0, 0.8, 1.0),
            (2, 1, 0.9, 1.0),
        )
        for repeat, projection_seed, score, c_value in settings:
            rows.append(
                {
                    "repeat": repeat,
                    "projection_seed": projection_seed,
                    "target": "footwear",
                    "condition": "pixels",
                    "family": "pixels",
                    "layer": "pixels",
                    "sample_size": 10,
                    "original_feature_count": 784,
                    "feature_count": 32,
                    "selected_c": c_value,
                    "validation_accuracy": score,
                    "validation_macro_f1": score,
                    "test_accuracy": score,
                    "test_macro_f1": score,
                    "test_target_f1": score,
                }
            )
        aggregate = experiment.aggregate_results(rows)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["n_sampling_repeats"], 2)
        self.assertEqual(aggregate[0]["n_projection_seeds"], 2)
        self.assertEqual(aggregate[0]["n_runs"], 4)
        self.assertAlmostEqual(aggregate[0]["test_macro_f1_mean"], 0.75)
        self.assertAlmostEqual(
            aggregate[0]["test_macro_f1_std"],
            np.std([0.6, 0.7, 0.8, 0.9], ddof=1),
        )

    def test_run_fits_every_probe_before_test_evaluation(self) -> None:
        events = []
        labels = np.tile(np.arange(10), 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"data_dir": "data", "probe_cs": [0.1], "probe_max_iter": 50}),
                encoding="utf-8",
            )
            np.savez_compressed(
                source / "split_indices.npz",
                cnn_train=np.arange(30),
                cnn_validation=np.arange(30, 40),
                probe_train=np.arange(20),
                probe_validation=np.arange(30, 40),
                test=np.arange(10),
            )
            (source / "random_cnn.pt").write_bytes(b"random")
            (source / "trained_cnn_best.pt").write_bytes(b"trained")
            config = experiment.SampleEfficiencyConfig(
                source_dir=source,
                output_dir=output,
                data_dir=None,
                seed=42,
                sample_sizes=(10,),
                repeats=1,
                feature_batch_size=8,
                num_workers=0,
                requested_device="cpu",
                threads=1,
            )

            def fake_fit(*args, **kwargs):
                events.append("fit")
                return [], {(0, 10): np.arange(10)}

            def fake_evaluate(*args, **kwargs):
                events.append("test")
                self.assertEqual(events, ["fit", "test"])
                return []

            with (
                patch.object(experiment, "fit_all_probes", side_effect=fake_fit),
                patch.object(
                    experiment,
                    "evaluate_fitted_probes",
                    side_effect=fake_evaluate,
                ),
                patch.object(experiment, "save_results_csv"),
                patch.object(experiment, "save_sample_efficiency_plot"),
            ):
                experiment.run(config, TinyImages(labels), TinyImages(labels))
            self.assertEqual(events, ["fit", "test"])
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stage"], "complete")
            self.assertFalse(status["cnn_retrained"])
            source_manifest = json.loads(
                (output / "source_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(source_manifest["source_probe_results_used"])


if __name__ == "__main__":
    unittest.main()
