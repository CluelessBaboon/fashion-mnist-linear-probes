"""CNN architecture and frozen representation conditions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import (
    CONV1_CHANNELS,
    CONV2_CHANNELS,
    CONV_KERNEL_SIZE,
    CONV_PADDING,
    HIDDEN_UNITS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    NUM_CLASSES,
    POOL_SIZE,
)


@dataclass
class FeatureCondition:
    """Source of features supplied to a linear probe.

    Attributes:
        family: Representation family: pixels, random, or trained.
        layer: Pixel input or hidden layer: pixels, conv1, conv2, or fc.
        model: CNN supplying hidden activations; None for raw pixels.
    """

    family: str
    layer: str
    model: nn.Module | None

    @property
    def name(self) -> str:
        """Return the stable condition identifier used in saved results."""
        if self.family == "pixels":
            return "pixels"
        return f"{self.family}_{self.layer}"


class SmallCNN(nn.Module):
    """Two convolutional stages followed by a 128-unit representation.

    Inputs have shape (batch, 1, 28, 28). Both convolutional stages apply
    ReLU followed by 2x2 max pooling. The classification head returns ten
    logits for cross-entropy training.

    Layer attribute names match the original checkpoint state dictionaries.
    """

    def __init__(self) -> None:
        """Initialize layers in the original order and with original sizes."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            IMAGE_CHANNELS,
            CONV1_CHANNELS,
            kernel_size=CONV_KERNEL_SIZE,
            stride=1,
            padding=CONV_PADDING,
        )
        self.conv2 = nn.Conv2d(
            CONV1_CHANNELS,
            CONV2_CHANNELS,
            kernel_size=CONV_KERNEL_SIZE,
            stride=1,
            padding=CONV_PADDING,
        )
        self.pool = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
        pooled_size = IMAGE_SIZE // (POOL_SIZE**2)
        self.fc = nn.Linear(
            CONV2_CHANNELS * pooled_size * pooled_size,
            HIDDEN_UNITS,
        )
        self.classifier = nn.Linear(HIDDEN_UNITS, NUM_CLASSES)

    def forward(self, images: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute classification logits and intermediate representations.

        Args:
            images: Float image tensor with shape (batch, 1, 28, 28).

        Returns:
            Logits with shape (batch, 10), followed by activations keyed by:
            conv1: (batch, 16, 14, 14);
            conv2: (batch, 32, 7, 7);
            fc: (batch, 128).
        """
        conv1_features = self.pool(torch.relu(self.conv1(images)))
        conv2_features = self.pool(torch.relu(self.conv2(conv1_features)))
        fc_features = torch.relu(
            self.fc(torch.flatten(conv2_features, start_dim=1))
        )
        logits = self.classifier(fc_features)
        activations = {
            "conv1": conv1_features,
            "conv2": conv2_features,
            "fc": fc_features,
        }
        return logits, activations


def state_dict_on_cpu(model: nn.Module) -> dict[str, Tensor]:
    """Copy model parameters and buffers into an independent CPU state.

    Args:
        model: Module whose state should be copied.

    Returns:
        State dictionary containing detached, cloned CPU tensors.
    """
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def freeze_model(model: nn.Module) -> None:
    """Disable parameter gradients and switch a model to evaluation mode.

    Args:
        model: Module to modify in place.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)