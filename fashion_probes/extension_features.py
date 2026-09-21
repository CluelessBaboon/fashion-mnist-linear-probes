"""Batched frozen feature extraction that retains original class labels."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .extension_config import LAYERS


@torch.no_grad()
def extract_features(
    model: nn.Module | None, loader: DataLoader, device: torch.device
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract all hidden layers in one pass, or raw pixels for model=None.

    Args:
        model: Frozen CNN; None selects raw pixels.
        loader: Nonempty unshuffled loader with original dataset labels.
        device: Feature-extraction device.

    Returns:
        Float32 feature arrays and original labels in identical row order.

    Notes:
        Arrays are preallocated to avoid concatenation copies. A supplied
        model is returned to the CPU after extraction. Label conversion is
        deliberately deferred to each downstream probe task.
    """
    if model is not None:
        model.to(device)
        model.eval()
    features = {}
    labels = np.empty(len(loader.dataset), dtype=np.int64)
    offset = 0
    for images, original in loader:
        images = images.to(device, non_blocking=True)
        if model is None:
            batches = {"pixels": images.flatten(1)}
        else:
            _, activations = model(images)
            batches = {layer: activations[layer].flatten(1) for layer in LAYERS}
        end = offset + len(images)
        for layer, batch in batches.items():
            values = batch.detach().cpu().numpy()
            if layer not in features:
                features[layer] = np.empty(
                    (len(loader.dataset), values.shape[1]), dtype=np.float32
                )
            features[layer][offset:end] = values
        labels[offset:end] = original.numpy()
        offset = end
    if model is not None:
        model.cpu()
    if offset != len(loader.dataset) or not features:
        raise ValueError("Feature extraction requires a nonempty complete loader.")
    return features, labels
