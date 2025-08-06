from .optimizer import Optimizer
import numpy as np
from typing import Tuple


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if group["weight_decay"] != 0:
                    grad = grad + group["weight_decay"] * param.data

                state = self.state.setdefault(param, {})

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(param.data)
                    state["exp_avg_sq"] = np.zeros_like(param.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad**2

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1

                denom = np.sqrt(exp_avg_sq / bias_correction2) + group["eps"]
                param.data = param.data - step_size * exp_avg / denom
