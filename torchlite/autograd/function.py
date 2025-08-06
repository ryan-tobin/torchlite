"""
Base class for all differentiable functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class FunctionCtx:
    """Context object to pass information between forward and backward."""

    def __init__(self):
        self.saved_tensors = []
        self.needs_input_grad = []

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors):
        """Mark tensors as non-differentiable."""
        for tensor in tensors:
            tensor.requires_grad = False


class Function(ABC):
    """
    Base class for creating custom autograd functions.

    To create a custom function, subclass this and implement:
    - forward(ctx, *args, **kwargs): Compute the forward pass
    - backward(ctx, *grad_outputs): Compute gradients
    """

    @staticmethod
    @abstractmethod
    def forward(ctx: FunctionCtx, *args, **kwargs):
        """
        Perform the forward pass.

        Args:
            ctx: Context object for saving information
            *args: Function inputs

        Returns:
            Function output(s)
        """
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx: FunctionCtx, *grad_outputs):
        """
        Compute gradients with respect to inputs.

        Args:
            ctx: Context object with saved information
            *grad_outputs: Gradients with respect to outputs

        Returns:
            Gradients with respect to each input
        """
        pass

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the function."""
        ctx = FunctionCtx()
        outputs = cls.forward(ctx, *args, **kwargs)

        if any(getattr(arg, "requires_grad", False) for arg in args):
            outputs._backward_fn = lambda grad: cls.backward(ctx, grad)

        return outputs


class Exp(Function):
    """Exponential function."""

    @staticmethod
    def forward(ctx, input):
        output = np.exp(input.data)
        ctx.save_for_backward(output)
        return type(input)(output, requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return grad_output * output


class Log(Function):
    """Natural logarithm function."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return type(input)(
            np.log(
                input.data),
            requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / input.data


class Pow(Function):
    """Power function."""

    @staticmethod
    def forward(ctx, input, exponent):
        ctx.save_for_backward(input, exponent)
        output = np.power(input.data, exponent)
        return type(input)(output, requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        input, exponent = ctx.saved_tensors
        return grad_output * exponent * \
            np.power(input.data, exponent - 1), None
