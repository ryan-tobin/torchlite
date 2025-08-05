"""Pooling layers for neural networks."""
import numpy as np 
from .module import Module 
from ..tensor import Tensor
from typing import Tuple 

class MaxPool2d(Module):
    """2D max pooling layer."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x_padded = np.pad(x.data, ((0,0), (0,0), (ph, ph), (pw,pw)), mode='constant', constant_values=-np.inf)
        else:
            x_padded = x.data 
        
        out_h = (height + 2 * ph - kh) // sh + 1
        out_w = (width + 2 * pw - kw) // sw + 1

        output = np.zeros((batch_size, channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh 
                w_start = j * sw 
                h_end = h_start + kh 
                w_end = w_start + kw 

                pool_region = x_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(pool_region, axis=(2,3))

        return Tensor(output, requires_grad=x.requires_grad)
    