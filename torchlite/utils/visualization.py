"""Visualization utilities for neural networks."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..nn.module import Module


class GradientFlow:
    """Visualize gradient flow through the network during training."""

    def __init__(self, model: Module):
        self.model = model
        self.gradient_stats = []

    def plot(self):
        """Plot gradient flow."""
        ave_grads = []
        max_grads = []
        layers = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layers.append(name)
                ave_grads.append(np.mean(np.abs(param.grad)))
                max_grads.append(np.max(np.abs(param.grad)))

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, label="max gradient")
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.6, label="mean gradient")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="black")
        plt.xticks(range(0, len(ave_grads)), layers, rotation="vertical")
        plt.xlim(left=-1, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("Gradient magnitude")
        plt.title("Gradient flow")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves."""

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.show()
    else:
        plt.show


def visualize_activations(
    activations: Dict[str, np.ndarray], save_path: Optional[str] = None
) -> None:
    """ "Visualize layer activiation"""

    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    for ax, (layer_name, activation) in zip(axes, activations.items()):
        if len(activation.shape) == 4:
            img = activation[0, 0]
        elif len(activation.shape) == 2:
            img = activation[0].reshape(-1, 1)
        else:
            continue

        ax.imshow(img, cmap="viridis")
        ax.set_title(layer_name)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.show()
    else:
        plt.show()
