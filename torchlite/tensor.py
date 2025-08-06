import numpy as np
from typing import Union, Tuple, List, Optional
import weakref


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support.
    This is the core data structure of TorchLite.
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float],
        requires_grad: bool = False,
        _children: Tuple = (),
        _op: str = "",
    ):
        if isinstance(data, (list, float, int)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

        self._version = 0

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            if self.requires_grad:
                # Handle broadcasting: sum over broadcasted dimensions
                grad = out.grad
                # Sum over dimensions that were broadcast
                ndims_added = len(grad.shape) - len(self.shape)
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)

                # Sum over dimensions that were size 1
                for i, (grad_dim, data_dim) in enumerate(zip(grad.shape, self.shape)):
                    if data_dim == 1 and grad_dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad = self.grad + grad

            if other.requires_grad:
                # Handle broadcasting: sum over broadcasted dimensions
                grad = out.grad
                # Sum over dimensions that were broadcast
                ndims_added = len(grad.shape) - len(other.shape)
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)

                # Sum over dimensions that were size 1
                for i, (grad_dim, data_dim) in enumerate(zip(grad.shape, other.shape)):
                    if data_dim == 1 and grad_dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = grad
                else:
                    other.grad = other.grad + grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = other.data * out.grad
                else:
                    self.grad = self.grad + other.data * out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = self.data * out.grad
                else:
                    other.grad = other.grad + self.data * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiplication with automatic broadcasting."""
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@",
        )

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad @ other.data.T
                else:
                    self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = self.data.T @ out.grad
                else:
                    other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="ReLU",
        )

        def _backward():
            if self.requires_grad:
                self.grad = (
                    self.grad + (self.data > 0) * out.grad
                    if self.grad is not None
                    else (self.data > 0) * out.grad
                )

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, self.shape)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, self.shape) / self.data.size
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def backward(self, gradient=None):
        """Compute gradients using reverse-mode automatic differentiation"""
        if gradient is None:
            gradient = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = gradient
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradients to None."""
        self.grad = None

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="-",
        )

        def _backward():
            if self.requires_grad:
                # Handle broadcasting
                grad = out.grad
                ndims_added = len(grad.shape) - len(self.shape)
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)

                for i, (grad_dim, data_dim) in enumerate(zip(grad.shape, self.shape)):
                    if data_dim == 1 and grad_dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad = self.grad + grad

            if other.requires_grad:
                # Handle broadcasting with negation
                grad = -out.grad
                ndims_added = len(grad.shape) - len(other.shape)
                for i in range(ndims_added):
                    grad = grad.sum(axis=0)

                for i, (grad_dim, data_dim) in enumerate(zip(grad.shape, other.shape)):
                    if data_dim == 1 and grad_dim > 1:
                        grad = grad.sum(axis=i, keepdims=True)

                if other.grad is None:
                    other.grad = grad
                else:
                    other.grad = other.grad + grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        """Right subtraction (for scalar - tensor)."""
        return Tensor(other) - self

    def __neg__(self):
        """Negation operator."""
        return self * -1

    def __truediv__(self, other):
        """Division operator."""
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="/",
        )

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = out.grad / other.data
                else:
                    self.grad = self.grad + out.grad / other.data
            if other.requires_grad:
                grad = -out.grad * self.data / (other.data**2)
                if other.grad is None:
                    other.grad = grad
                else:
                    other.grad = other.grad + grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        """Right division (for scalar / tensor)."""
        return Tensor(other) / self

    def __pow__(self, power):
        """Power operator."""
        assert isinstance(power, (int, float)), "Only supporting int/float powers for now"
        out = Tensor(
            self.data**power, requires_grad=self.requires_grad, _children=(self,), _op=f"**{power}"
        )

        def _backward():
            if self.requires_grad:
                grad = power * (self.data ** (power - 1)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """Reshape tensor."""
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.shape)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def transpose(self, axis1, axis2):
        """Transpose tensor."""
        out = Tensor(
            np.transpose(
                self.data,
                [
                    axis2 if i == axis1 else axis1 if i == axis2 else i
                    for i in range(len(self.shape))
                ],
            ),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward():
            if self.requires_grad:
                # Transpose back
                grad = np.transpose(
                    out.grad,
                    [
                        axis2 if i == axis1 else axis1 if i == axis2 else i
                        for i in range(len(out.grad.shape))
                    ],
                )
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent activation."""
        out = Tensor(
            np.tanh(self.data), requires_grad=self.requires_grad, _children=(self,), _op="tanh"
        )

        def _backward():
            if self.requires_grad:
                grad = (1 - out.data**2) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def _unbroadcast(grad, shape):
        """
        Sum out broadcasted dimensions to match original shape.
        """
        # Handle scalar shapes
        if not shape:
            return np.sum(grad)

        # Sum over extra leading dimensions
        ndims_added = len(grad.shape) - len(shape)
        for i in range(ndims_added):
            grad = grad.sum(axis=0)

        # Sum over dimensions that were size 1
        for i, (grad_dim, data_dim) in enumerate(zip(grad.shape, shape)):
            if data_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad
