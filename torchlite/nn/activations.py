from .module import Module 
from ..tensor import Tensor 
import numpy as np 

class ReLU(Module):
    """Rectified Linear Unit activation."""
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
class Sigmoid(Module):
    """Sigmoid activiation."""
    def forward(self, x: Tensor) -> Tensor:
        data = 1 / (1 + np.exp(-x.data))
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op='sigmoid')

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + data * (1 - data) * out.grad if x.grad is not None else data * (1 - data) * out.grad

        out._backward = _backward
        return out
    
class Tanh(Module):
    """Hyperbolic tangent activation."""
    def forward(self, x: Tensor) -> Tensor:
        data = np.tanh(x.data)
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,),_op='tanh')

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + (1 - data**2) * out.grad if x.grad is not None else (1 - data**2) * out.grad 

        out._backward = _backward
        return out 
    
class Softmax(Module):
    """Softmax activation."""
    def __init__(self, dim: int = -1):
        super().__init_()
        self.dim = dim 
    
    def forward(self, x: Tensor) -> Tensor:
        exp_x = np.exp(x.data - np.max(x.data, axis=self.dim, keepdims=True))
        data = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        out = Tensor(data, requires_grad=x.requires_grad, _children=(x,), _op='softmax')

        def _backward():
            if x.requires_grad:
                s = data.reshape(-1, 1)
                jacobian = np.diagflat(s) - np.dot(s, s.T)
                x.grad = x.grad + np.dot(jacobian, out.grad.reshape(-1,1)).reshape(x.shape) if x.grad is not None else np.dot(jacobian, out.grad.reshape(-1,1)).reshape(x.shape)

        out._backward = _backward
        return out 
    
class LeakyReLU(Module):
    """Leaky ReLU Activation."""

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        data = np.where(x.data > 0, x.data, x.data * self.negative_slope)
        return Tensor(data, requires_grad=x.requires_grad)
    
class SiLU(Module):
    """Sigmoid Linear Unit (Swish) activation"""

    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x.data))
        data = x.data * sigmoid 
        return Tensor(data, requires_grad=x.requires_grad)

class GELU(Module):
    """Guassian Error Linear Unit activation."""

    def forward(self, x):
        data = 0.5 * x.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data**3)))
        return Tensor(data, requires_grad=x.requires_grad)