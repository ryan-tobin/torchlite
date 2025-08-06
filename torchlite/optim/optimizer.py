from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..nn.module import Parameter


class Optimizer(ABC):
    """Base class for all optimizers"""

    def __init__(self, params: List[Parameter], defaults: Dict[str, Any]):
        self.param_groups = [{"params": list(params), **defaults}]
        self.state = {}

    @abstractmethod
    def step(self):
        """Perform a single optimization step."""
        pass

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for group in self.param_groups:
            for param in group["params"]:
                param.zero_grad()
