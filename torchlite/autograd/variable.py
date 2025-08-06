"""
Variable wrapper for tensors with autograd support.
Legacy API for compatibility.
"""

from ..tensor import Tensor
import warnings


class Variable(Tensor):
    """Legacy Variable class for backward compatibility.
    In modern TorchLite, Tensor has autograd built-in
    """

    def __init__(self, data, requires_grad=False, volatile=False):
        warnings.warn(
            "Variable is deprecated. Use Tensor with requires_grad=True instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(data, requires_grad=requires_grad)
        self.volatile = volatile
