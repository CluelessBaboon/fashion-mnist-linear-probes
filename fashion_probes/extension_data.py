"""Leakage-aware split construction and paired, nested probe sampling."""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from .extension_config import ExtensionConfig, Task


def stratified_subset(
    ids: np.ndarray, labels: np.ndarray, size: int, seed: int
) -> np.ndarray:
    """Select original IDs while retaining all ten original classes.

    Args:
        ids: Eligible original dataset indices.
        labels: Labels indexed by original dataset position.
        size: Requested subset size.
        seed: Sampling seed.

    Returns:
        Selected original indices.

    Raises:
        ValueError: If the subset cannot contain all ten classes.
    """
    if size == len(ids):
        result = ids.copy()
    else:
        result, _ = train_test_split(
            ids, train_size=size, stratify=labels[ids], random_state=seed
        )
    if set(labels[result].tolist()) != set(range(10)):
        raise ValueError("Every split must contain all ten original classes.")
    return np.asarray(result, dtype=np.int64)


def make_splits(
    train_labels: np.ndarray, test_labels: np.ndarray, config: ExtensionConfig
) -> dict[str, np.ndarray]:
    """Create one fixed split shared by every initialization seed.

    Args:
        train_labels: Labels of the official training dataset.
        test_labels: Labels of the separate official test dataset.
        config: Split sizes, seed, and previous test-index archives.

    Returns:
        Original IDs for CNN train/validation, probe validation, test, exclusions.

    Raises:
        ValueError: If sizes or exclusion archives make the split invalid.
    """
    total = config.cnn_train_size + config.cnn_val_size
    if total > len(train_labels):
        raise ValueError("Requested CNN split exceeds the available training dataset.")
    pool = stratified_subset(
        np.arange(len(train_labels)), train_labels, total, config.split_seed
    )
    train, validation = train_test_split(
        pool,
        train_size=config.cnn_train_size,
        test_size=config.cnn_val_size,
        stratify=train_labels[pool],
        random_state=config.split_seed + 1,
    )
    probe_validation = stratified_subset(
        validation, train_labels, config.probe_val_size, config.split_seed + 3
    )
    excluded_parts = []
    for path in config.exclude_test_indices:
        with np.load(path, allow_pickle=False) as archive:
            if "test" not in archive:
                raise ValueError(f"{path} has no 'test' index array.")
            excluded = np.asarray(archive["test"])
        if excluded.ndim != 1 or not np.issubdtype(excluded.dtype, np.integer):
            raise ValueError(
                f"{path}: test IDs must be a one-dimensional integer array."
            )
        if np.any(excluded < 0) or np.any(excluded >= len(test_labels)):
            raise ValueError(f"{path}: test IDs are outside the official test dataset.")
        excluded_parts.append(excluded)
    excluded = (
        np.unique(np.concatenate(excluded_parts))
        if excluded_parts
        else np.empty(0, dtype=np.int64)
    )
    available = np.setdiff1d(np.arange(len(test_labels)), excluded)
    test_size = config.test_size if config.test_size is not None else len(available)
    if test_size > len(available) or test_size < 10:
        raise ValueError(
            "Test exclusions leave too few examples for the requested test size."
        )
    test = stratified_subset(available, test_labels, test_size, config.split_seed + 4)
    if np.intersect1d(train, validation).size:
        raise RuntimeError("CNN training and validation splits overlap.")
    return {
        "cnn_train": train,
        "cnn_validation": validation,
        "probe_validation": probe_validation,
        "test": test,
        "excluded_test": excluded,
    }


def task_targets(
    original_labels: np.ndarray, task: Task
) -> tuple[np.ndarray, np.ndarray]:
    """Filter task-eligible examples and map labels to contiguous target IDs.

    Args:
        original_labels: Fashion-MNIST labels in feature-row order.
        task: Ordered original-label groups.

    Returns:
        Boolean row mask and target IDs for the masked rows.
    """
    mapped = np.full(len(original_labels), -1, dtype=np.int64)
    for target, group in enumerate(task.groups):
        mapped[np.isin(original_labels, group)] = target
    mask = mapped >= 0
    return mask, mapped[mask]


def make_probe_samples(
    train_ids: np.ndarray,
    labels: np.ndarray,
    tasks: list[Task],
    budgets: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Create shared feature-pool IDs and nested, balanced task samples.

    Original-class pools are shuffled once. A grouped class (non-footwear)
    interleaves its original classes, so their prefix counts differ by at most
    one. Budgets count target classes, not original classes within a group.

    Args:
        train_ids: CNN-training indices only.
        labels: Full original training labels.
        tasks: Prespecified probe definitions.
        budgets: Positive examples-per-target-class values.
        seed: Seed controlling probe sampling.

    Returns:
        Sorted union of all required original IDs and a dictionary mapping
        (task name, budget) to feature-row positions in that union.

    Raises:
        ValueError: If any group cannot supply the largest requested budget.
    """
    rng = np.random.default_rng(seed + 200)
    pools = {
        label: rng.permutation(train_ids[labels[train_ids] == label])
        for label in range(10)
    }
    group_pools = {}
    for task in tasks:
        for group in task.groups:
            if group in group_pools:
                continue
            order = rng.permutation(group)
            depth = min(len(pools[int(label)]) for label in order)
            group_pools[group] = np.stack(
                [pools[int(label)][:depth] for label in order],
                axis=1,
            ).reshape(-1)
            if len(group_pools[group]) < max(budgets):
                raise ValueError(f"Not enough training examples for group {group}.")
    selections = {
        (task.name, budget): np.concatenate(
            [group_pools[group][:budget] for group in task.groups]
        )
        for task in tasks
        for budget in budgets
    }
    pool_ids = np.unique(np.concatenate(list(selections.values())))
    positions = {key: np.searchsorted(pool_ids, ids) for key, ids in selections.items()}
    return pool_ids, positions


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's assigned worker seed."""
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: Dataset,
    ids: np.ndarray,
    config: ExtensionConfig,
    device: torch.device,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Build a loader with an independent, explicitly seeded generator.

    Args:
        dataset: Image/original-label dataset.
        ids: Original indices to load.
        config: Batch and worker settings.
        device: Device controlling pinned-memory use.
        shuffle: True only for CNN training loaders.
        seed: Local generator seed.

    Returns:
        Loader over the supplied indices.
    """
    return DataLoader(
        Subset(dataset, ids.tolist()),
        batch_size=config.batch_size if shuffle else config.feature_batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
