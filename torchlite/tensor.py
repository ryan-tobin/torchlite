import numpy as np 
from typing import Union, Tuple, List, Optional 
import weakref 

class Tensor:
    """
    A multi-dimensional array with automatic differentiation support.
    This is the core data structure of TorchLite.
    """
    def __init__(self, data: Union[np.ndarray, list, float],
                 requires_grad: bool = False,
                 _children: Tuple = (),
                 _op: str = ''):
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
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad - other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + other.data * out.grad if self.grad is not None else other.data * out.grad
            if other.requires_grad:
                other.grad = other.grad + self.data * out.grad if other.grad is not None else self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Matrix multiplication with automatic broadcasting."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='@')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T if self.grad is not None else out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad if other.grad is not None else self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='ReLU')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (self.data > 0) * out.grad if self.grad is not None else (self.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sum')
        
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
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='mean')
        
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