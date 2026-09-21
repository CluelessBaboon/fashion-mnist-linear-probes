"""Train paired fine/coarse CNNs while reusing the original model class."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .extension_config import COARSE_TARGETS, ExtensionConfig
from .extension_data import make_loader
from .extension_reporting import save_json, save_training_curves
from .models import SmallCNN


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch and request deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps, raising RuntimeError for unavailable devices."""
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if mps else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "mps" and not mps:
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(name)


def make_models(seed: int) -> dict[str, SmallCNN]:
    """Create independent CNNs with identical initial feature extractors.

    Args:
        seed: Initialization seed.

    Returns:
        Random ten-class, fine ten-class, and coarse two-class models.

    Notes:
        Only the extension's copied coarse model receives a new head. The
        original SmallCNN constructor and original experiment stay compatible.
    """
    seed_everything(seed)
    random_model = SmallCNN()
    fine = copy.deepcopy(random_model)
    coarse = copy.deepcopy(random_model)
    coarse.classifier = nn.Linear(random_model.classifier.in_features, 2)
    return {"random": random_model, "fine": fine, "coarse": coarse}


def freeze(model: nn.Module) -> None:
    """Set evaluation mode and disable parameter gradients in place."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def cpu_state(model: nn.Module) -> dict[str, Tensor]:
    """Return independent CPU copies of a model's parameters and buffers."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def cnn_targets(
    labels: Tensor, family: str, coarse_target: str = "footwear"
) -> Tensor:
    """Map original labels to the training objective for this CNN family."""
    if family != "coarse":
        return labels
    if coarse_target not in COARSE_TARGETS:
        raise ValueError(f"Unknown coarse target: {coarse_target!r}.")
    target = torch.zeros_like(labels, dtype=torch.bool)
    positive, _ = COARSE_TARGETS[coarse_target]
    for label in positive:
        target |= labels == label
    return target.long()


@torch.no_grad()
def evaluate_cnn(
    model: SmallCNN,
    loader: DataLoader,
    device: torch.device,
    family: str,
    coarse_target: str = "footwear",
) -> dict[str, float]:
    """Return sample-mean cross-entropy and accuracy on the CNN's own task.

    Args:
        model: Model already on the evaluation device.
        loader: Nonempty image/original-label loader.
        device: Computation device.
        family: fine or coarse, controlling target mapping.
        coarse_target: Positive semantic group for the coarse family.

    Returns:
        loss and accuracy for the selected CNN task.
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        targets = cnn_targets(labels, family, coarse_target).to(device, non_blocking=True)
        logits, _ = model(images)
        loss_sum += criterion(logits, targets).item()
        correct += (logits.argmax(1) == targets).sum().item()
        count += len(targets)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def train_cnn(
    model: SmallCNN,
    family: str,
    dataset: Dataset,
    splits: dict,
    config: ExtensionConfig,
    device: torch.device,
    seed: int,
    directory: Path,
) -> list[dict]:
    """Train a CNN for the fixed epoch budget and restore its best checkpoint.

    Fresh loaders use the same seed for fine and coarse training, producing
    matching minibatch orders. Adam, constant learning rate, and cross-entropy
    match the original experiment; no augmentation is added.

    Args:
        model: Trainable fine or coarse model.
        family: Name of its supervision condition.
        dataset: Official training dataset.
        splits: Fixed train and validation indices.
        config: Training settings.
        device: Training device.
        seed: Paired loader seed base.
        directory: Existing seed output directory.

    Returns:
        Epoch history, also written to JSON and plotted.

    Raises:
        RuntimeError: If a loss becomes nonfinite or no checkpoint is selected.
    """
    model.to(device)
    training = make_loader(
        dataset, splits["cnn_train"], config, device, True, seed + 100
    )
    validation = make_loader(dataset, splits["cnn_validation"], config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for images, labels in training:
            images = images.to(device, non_blocking=True)
            targets = cnn_targets(labels, family, config.coarse_target).to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {family} training loss.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            count += len(targets)
        scores = evaluate_cnn(
            model, validation, device, family, config.coarse_target
        )
        if not np.isfinite(scores["loss"]):
            raise RuntimeError(f"Nonfinite {family} validation loss.")
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "train_accuracy": correct / count,
                "validation_loss": scores["loss"],
                "validation_accuracy": scores["accuracy"],
            }
        )
        print(
            f"  {family} epoch {epoch}/{config.epochs}: "
            f"train accuracy={correct / count:.4f}, validation accuracy={scores['accuracy']:.4f}"
        )
        if scores["loss"] < best_loss:
            best_loss, best_epoch = scores["loss"], epoch
            best_state = cpu_state(model)
            torch.save(best_state, directory / f"{family}_cnn.pt")
        save_json(directory / f"{family}_history.json", history)
    if best_state is None:
        raise RuntimeError("No CNN checkpoint was selected.")
    model.load_state_dict(best_state)
    model.cpu()
    freeze(model)
    save_json(
        directory / f"{family}_selection.json",
        {
            "best_epoch": best_epoch,
            "validation_loss": best_loss,
            "epochs_completed": len(history),
        },
    )
    save_training_curves(history, directory / f"{family}_training")
    return history
