import numpy as np 
from .module import Module, Parameter
from ..tensor import Tensor 
from typing import Optional, Tuple, Union 

class Linear(Module):
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        )
        if bias:
            self.bias = Parameter(np.zeros((1, out_features)))
        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias 
        return out 
    
class Conv2D(Module):
    """2D Convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride 
        self.padding = (padding, padding) if isinstance(padding, int) else padding 

        k_h, k_w = self.kernel_size
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, k_h, k_w) *
            np.sqrt(2.0 / (in_channels * k_h * k_w))
        )
        if bias:
            self.bias = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None 

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError

class Dropout(Module):
    """Dropout layer for regularization."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
            return x * Tensor(mask / (1 - self.p))
        return x

class BatchNorm1d(Module):
    """1D Batch Normalization"""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps 
        self.momentum = momentum

        self.weight = Parameter(np.ones((num_features,)))
        self.bias = Parameter(np.zeros((num_features,)))

        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_mean = np.mean(x.data, axis=0)
            batch_var = np.var(x.data, axis=0)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            x_norm = (x.data - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            x_norm = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        output = x_norm * self.weight.data + self.bias.data 
        return Tensor(output, requires_grad=x.requires_grad or self.weight.requires_grad)
    
class BatchNorm2d(Module):
    """2D Batch Normalization"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps 
        self.momentum = momentum

        self.weight = Parameter(np.ones((num_features, 1, 1)))
        self.bias = Parameter(np.zeroes((num_features, 1, 1)))

        self.running_mean = np.zeros((num_features, 1, 1))
        self.running_var = np.ones((num_features, 1, 1))

    def forward(self, x):
        if self.training:
            mean = x.data.mean(axis=(0,2,3), keepdims=True)
            var = x.data.var(axis=(0,2,3), keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var 

            x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)

        output = x_norm * self.weight.data + self.bias.data 
        return Tensor(output, requires_grad=x.requires_grad or self.weight.requires_grad)
    
class LayerNorm(Module):
    """Layer Normalization"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps 

        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))

    def forward(self, x):
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        output = x_norm * self.weight.data + self.bias.data 
        return Tensor(output, requires_grad=x.requires_grad or self.weight.requires_grad)
    
