"""Dataset preparation, stratified sampling, and footwear target mapping."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import Config, FOOTWEAR_CLASSES, TOP_CLASSES


@dataclass
class DataBundle:
    """Data loaders and the original dataset indices underlying each split.

    Attributes:
        cnn_train: Shuffled CNN training loader.
        cnn_validation: Unshuffled CNN validation loader.
        probe_train: Unshuffled subset of the CNN training split.
        probe_validation: Unshuffled subset of the CNN validation split.
        test: Unshuffled official test-set loader or stratified subset.
        split_indices: Original indices used to construct the five loaders.
    """

    cnn_train: DataLoader
    cnn_validation: DataLoader
    probe_train: DataLoader
    probe_validation: DataLoader
    test: DataLoader
    split_indices: dict[str, np.ndarray]


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a PyTorch data-loader worker.

    Args:
        worker_id: Worker identifier supplied by PyTorch. PyTorch's assigned
            worker seed already incorporates this identifier.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stratified_train_validation_indices(
    labels: np.ndarray,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct disjoint splits while preserving original class proportions.

    Args:
        labels: Original class labels for the source dataset.
        train_size: Number of training examples.
        val_size: Number of validation examples.
        seed: Random seed for selecting and splitting the candidate pool.

    Returns:
        Training and validation arrays containing original dataset indices.

    Raises:
        ValueError: If the requested stratified split cannot be constructed.
    """
    all_indices = np.arange(len(labels))
    requested_total = train_size + val_size

    if requested_total < len(all_indices):
        selected_indices, _ = train_test_split(
            all_indices,
            train_size=requested_total,
            random_state=seed,
            stratify=labels,
        )
    else:
        selected_indices = all_indices

    train_indices, val_indices = train_test_split(
        selected_indices,
        train_size=train_size,
        test_size=val_size,
        random_state=seed + 1,
        stratify=labels[selected_indices],
    )
    return np.asarray(train_indices), np.asarray(val_indices)


def stratified_subset(
    available_indices: np.ndarray,
    labels: np.ndarray,
    subset_size: int,
    seed: int,
) -> np.ndarray:
    """Select a subset stratified by the original Fashion-MNIST classes.

    Args:
        available_indices: Original indices eligible for selection.
        labels: Class labels indexed by original dataset position.
        subset_size: Requested number of examples.
        seed: Random seed for subset selection.

    Returns:
        Selected original indices. Requesting the entire pool returns a copy.

    Raises:
        ValueError: If the requested stratified subset is infeasible.
    """
    if subset_size == len(available_indices):
        return np.asarray(available_indices).copy()

    selected_indices, _ = train_test_split(
        available_indices,
        train_size=subset_size,
        random_state=seed,
        stratify=labels[available_indices],
    )
    return np.asarray(selected_indices)


def make_loader(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build a seeded loader over selected original dataset indices.

    Args:
        dataset: Dataset returning image and class-label pairs.
        indices: Original indices to include.
        batch_size: Maximum examples per batch.
        shuffle: Whether to randomize example order.
        seed: Seed for the loader's independent random generator.
        num_workers: Number of background loading processes.
        pin_memory: Whether to allocate pinned host memory for CUDA transfers.

    Returns:
        Configured loader over the selected dataset subset.

    Raises:
        ValueError: If a data-loader option is invalid.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def prepare_data(config: Config, device: torch.device) -> DataBundle:
    """Download Fashion-MNIST, construct splits, and save their indices.

    ToTensor maps image pixels to float tensors in [0, 1]. Probe training
    samples come only from the CNN training split; probe validation samples
    come only from the CNN validation split.

    Args:
        config: Validated settings. Its output directory must already exist.
        device: Execution device, used to select pinned-memory loading.

    Returns:
        Loaders and indices for CNN training, validation, probing, and testing.

    Raises:
        ValueError: If a requested stratified split cannot be constructed.
        OSError: If dataset or split files cannot be read or written.
    """
    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    full_test = datasets.FashionMNIST(
        root=config.data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    train_labels = np.asarray(full_train.targets, dtype=np.int64)
    test_labels = np.asarray(full_test.targets, dtype=np.int64)

    cnn_train_indices, cnn_val_indices = stratified_train_validation_indices(
        train_labels,
        train_size=config.cnn_train_size,
        val_size=config.cnn_val_size,
        seed=config.seed,
    )
    probe_train_indices = stratified_subset(
        cnn_train_indices,
        train_labels,
        subset_size=config.probe_train_size,
        seed=config.seed + 2,
    )
    probe_val_indices = stratified_subset(
        cnn_val_indices,
        train_labels,
        subset_size=config.probe_val_size,
        seed=config.seed + 3,
    )

    all_test_indices = np.arange(len(test_labels))
    if config.test_size is None or config.test_size == len(all_test_indices):
        test_indices = all_test_indices
    else:
        test_indices = stratified_subset(
            all_test_indices,
            test_labels,
            subset_size=config.test_size,
            seed=config.seed + 4,
        )

    indices = {
        "cnn_train": cnn_train_indices,
        "cnn_validation": cnn_val_indices,
        "probe_train": probe_train_indices,
        "probe_validation": probe_val_indices,
        "test": test_indices,
    }
    np.savez_compressed(config.output_dir / "split_indices.npz", **indices)

    pin_memory = device.type == "cuda"
    return DataBundle(
        cnn_train=make_loader(
            full_train,
            cnn_train_indices,
            config.batch_size,
            shuffle=True,
            seed=config.seed + 10,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        cnn_validation=make_loader(
            full_train,
            cnn_val_indices,
            config.feature_batch_size,
            shuffle=False,
            seed=config.seed + 11,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        probe_train=make_loader(
            full_train,
            probe_train_indices,
            config.feature_batch_size,
            shuffle=False,
            seed=config.seed + 12,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        probe_validation=make_loader(
            full_train,
            probe_val_indices,
            config.feature_batch_size,
            shuffle=False,
            seed=config.seed + 13,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        test=make_loader(
            full_test,
            test_indices,
            config.feature_batch_size,
            shuffle=False,
            seed=config.seed + 14,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        split_indices=indices,
    )


def labels_to_footwear(original_labels: Tensor) -> Tensor:
    """Map sandal, sneaker, and ankle-boot labels to binary footwear targets.

    Args:
        original_labels: Tensor of original Fashion-MNIST class identifiers.

    Returns:
        Int64 tensor of the same shape and device: footwear is 1, otherwise 0.
    """
    footwear = torch.zeros_like(original_labels, dtype=torch.bool)
    for class_id in FOOTWEAR_CLASSES:
        footwear |= original_labels == class_id
    return footwear.to(torch.int64)


def labels_to_tops(original_labels: Tensor) -> Tensor:
    """Map T-shirt/top, pullover, coat, and shirt to binary top targets.

    Args:
        original_labels: Tensor of original Fashion-MNIST class identifiers.

    Returns:
        Int64 tensor of the same shape and device: top is 1, otherwise 0.
    """
    tops = torch.zeros_like(original_labels, dtype=torch.bool)
    for class_id in TOP_CLASSES:
        tops |= original_labels == class_id
    return tops.to(torch.int64)


def collect_footwear_labels(loader: DataLoader) -> np.ndarray:
    """Collect binary targets in data-loader iteration order.

    Args:
        loader: Nonempty loader yielding CPU image and original-label batches.

    Returns:
        One-dimensional array containing one binary target per example.
    """
    batches: list[np.ndarray] = []
    for _, original_labels in loader:
        batches.append(labels_to_footwear(original_labels).numpy())
    return np.concatenate(batches)
