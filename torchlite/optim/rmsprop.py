import numpy as np

from .optimizer import Optimizer


class RMSprop(Optimizer):
    """RMSprop optimizer"""

    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if group["weight_decay"] != 0:
                    grad = grad + group["weight_decay"] * param.data

                state = self.state.setdefault(param, {})

                if "square_avg" not in state:
                    state["square_avg"] = np.zeros_like(param.data)

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                square_avg = alpha * square_avg + (1 - alpha) * grad**2
                state["square_avg"] = square_avg

                param.data -= group["lr"] * grad / (np.sqrt(square_avg) + group["eps"])
