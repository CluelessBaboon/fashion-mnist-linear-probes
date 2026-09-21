"""CNN optimization and validation-based checkpoint selection."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import Config
from .models import SmallCNN, state_dict_on_cpu


def evaluate_cnn(
    model: SmallCNN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate ten-class cross-entropy and accuracy without gradients.

    Args:
        model: CNN already placed on the specified device.
        loader: Nonempty loader of images and original class labels.
        device: Device used for the forward pass.

    Returns:
        Mean per-example loss and accuracy.

    Notes:
        The model is left in evaluation mode.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, class_labels in loader:
            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            logits, _ = model(images)

            total_loss += criterion(logits, class_labels).item()
            total_correct += (logits.argmax(dim=1) == class_labels).sum().item()
            total_examples += class_labels.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_cnn(
    model: SmallCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Train with Adam and restore the lowest-validation-loss checkpoint.

    Args:
        model: Trainable CNN already placed on the specified device.
        train_loader: Nonempty training loader.
        val_loader: Nonempty validation loader.
        config: Optimization settings and checkpoint destination.
        device: Device used for optimization and validation.

    Returns:
        Per-epoch training and validation loss and accuracy.

    Raises:
        RuntimeError: If no valid checkpoint was selected.
        OSError: If the final checkpoint cannot be saved.

    Notes:
        The original fixed learning rate and epoch budget are preserved.
        The selected state is saved as trained_cnn_best.pt after training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, class_labels in train_loader:
            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = criterion(logits, class_labels)
            loss.backward()
            optimizer.step()

            batch_size = class_labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == class_labels).sum().item()
            total_examples += batch_size

        validation = evaluate_cnn(model, val_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{config.epochs}: "
            f"train loss={row['train_loss']:.4f}, "
            f"train accuracy={row['train_accuracy']:.4f}, "
            f"validation loss={row['validation_loss']:.4f}, "
            f"validation accuracy={row['validation_accuracy']:.4f}"
        )

        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_state = state_dict_on_cpu(model)

    if best_state is None:
        raise RuntimeError("Training ended without producing a CNN checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, config.output_dir / "trained_cnn_best.pt")
    return history