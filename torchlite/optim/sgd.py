import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0, weight_decay: float = 0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                d_p = param.grad

                if weight_decay != 0:
                    d_p = d_p + weight_decay * param.data

                if momentum != 0:
                    param_state = self.state.setdefault(param, {})
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = np.zeros_like(param.data)
                    else:
                        buf = param_state["momentum_buffer"]

                    buf = momentum * buf + d_p
                    d_p = buf

                param.data = param.data - group["lr"] * d_p
