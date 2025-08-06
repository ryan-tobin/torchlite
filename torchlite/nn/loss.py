import numpy as np

from ..tensor import Tensor
from .module import Module


class MSELoss(Module):
    """Mean Squared Error loss."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ((predictions - targets) ** 2).mean()


class CrossEntropyLoss(Module):
    """Cross-entropy loss with softmax."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        exp_logits = np.exp(
            logits.data -
            np.max(
                logits.data,
                axis=-
                1,
                keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if targets.data.ndim == 1:
            num_classes = logits.shape[-1]
            targets_onehot = np.eye(num_classes)[targets.data.astype(int)]
        else:
            targets_onehot = targets.data

        loss = -np.sum(targets_onehot * np.log(probs + 1e-8)) / logits.shape[0]

        out = Tensor(loss, requires_grad=logits.requires_grad,
                     _children=(logits,), _op="cross_entropy")

        def _backward():
            if logits.requires_grad:
                grad = (probs - targets_onehot) / logits.shape[0]
                logits.grad = (
                    logits.grad +
                    grad *
                    out.grad if logits.grad is not None else grad *
                    out.grad)

        out._backward = _backward
        return out


class BCELoss(Module):
    """Binary Cross Entropy Loss."""

    def forward(self, predictions, targets):
        eps = 1e-8
        loss = -(
            targets.data * np.log(predictions.data + eps)
            + (1 - targets.data) * np.log(1 - predictions.data + eps)
        )
        return Tensor(np.mean(loss), requires_grad=predictions.requires_grad)


class L1Loss(Module):
    """L1 (Mean Absolute Error) Loss."""

    def forward(self, predictions, targets):
        loss = np.abs(predictions.data - targets.data)
        return Tensor(np.mean(loss), requires_grad=predictions.requires_grad)
