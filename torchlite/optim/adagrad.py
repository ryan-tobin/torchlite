import numpy as np

from .optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer"""

    def __init__(self, params, lr=0.01, eps=1e-10, weight_decay=0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
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

                if "sum" not in state:
                    state["sum"] = np.zeros_like(param.data)

                state["sum"] += grad**2
                param.data -= group["lr"] * grad / \
                    (np.sqrt(state["sum"]) + group["eps"])
