from typing import Optional, Tuple, Union

import numpy as np

from ..tensor import Tensor
from .module import Module, Parameter


class Linear(Module):
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
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


class Conv2d(Module):
    """2D Convolutional layer with backward pass."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        # Initialize weights
        k_h, k_w = self.kernel_size
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, k_h, k_w)
            * np.sqrt(2.0 / (in_channels * k_h * k_w))
        )
        if bias:
            self.bias = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Store input for backward pass
        self.input = x

        batch_size, in_channels, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        # Pad input
        if p_h > 0 or p_w > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode="constant")
        else:
            x_padded = x.data

        # Store padded input for backward
        self.x_padded = x_padded

        # Calculate output dimensions
        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1

        # Perform convolution (simplified for clarity - production would use
        # im2col)
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * s_h
                        w_start = ow * s_w

                        # Extract patch and compute convolution
                        patch = x_padded[b, :, h_start : h_start + k_h, w_start : w_start + k_w]
                        output[b, oc, oh, ow] = np.sum(patch * self.weight.data[oc])

                        if self.bias is not None:
                            output[b, oc, oh, ow] += self.bias.data[oc]

        # Create output tensor with proper backward function
        out = Tensor(
            output,
            requires_grad=x.requires_grad or self.weight.requires_grad,
            _children=(x,),
            _op="Conv2d",
        )

        # Define backward pass
        def _backward():
            grad_output = out.grad
            batch_size, out_channels, out_h, out_w = grad_output.shape

            # Gradient w.r.t weight
            if self.weight.requires_grad:
                grad_weight = np.zeros_like(self.weight.data)

                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                h_start = oh * s_h
                                w_start = ow * s_w

                                patch = self.x_padded[
                                    b, :, h_start : h_start + k_h, w_start : w_start + k_w
                                ]
                                grad_weight[oc] += patch * grad_output[b, oc, oh, ow]

                if self.weight.grad is None:
                    self.weight.grad = grad_weight
                else:
                    self.weight.grad = self.weight.grad + grad_weight

            # Gradient w.r.t bias
            if self.bias is not None and self.bias.requires_grad:
                grad_bias = np.sum(grad_output, axis=(0, 2, 3))
                if self.bias.grad is None:
                    self.bias.grad = grad_bias
                else:
                    self.bias.grad = self.bias.grad + grad_bias

            # Gradient w.r.t input
            if x.requires_grad:
                grad_input = np.zeros_like(x.data)

                # Pad grad_input if necessary
                if p_h > 0 or p_w > 0:
                    grad_input_padded = np.pad(
                        grad_input, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode="constant"
                    )
                else:
                    grad_input_padded = grad_input

                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                h_start = oh * s_h
                                w_start = ow * s_w

                                grad_input_padded[
                                    b, :, h_start : h_start + k_h, w_start : w_start + k_w
                                ] += (self.weight.data[oc] * grad_output[b, oc, oh, ow])

                # Remove padding from gradient
                if p_h > 0 or p_w > 0:
                    grad_input = grad_input_padded[:, :, p_h:-p_h, p_w:-p_w]
                else:
                    grad_input = grad_input_padded

                if x.grad is None:
                    x.grad = grad_input
                else:
                    x.grad = x.grad + grad_input

        out._backward = _backward
        return out


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
    """1D Batch Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = Parameter(np.ones((num_features,)))
        self.bias = Parameter(np.zeros((num_features,)))

        # Running statistics (not parameters)
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x.data, axis=0)
            batch_var = np.var(x.data, axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_norm = (x.data - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            x_norm = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale and shift
        output = x_norm * self.weight.data + self.bias.data
        return Tensor(output, requires_grad=x.requires_grad or self.weight.requires_grad)


class BatchNorm2d(Module):
    """2D Batch Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(np.ones((num_features, 1, 1)))
        self.bias = Parameter(np.zeros((num_features, 1, 1)))

        self.running_mean = np.zeros((num_features, 1, 1))
        self.running_var = np.ones((num_features, 1, 1))

    def forward(self, x):
        if self.training:
            # N, C, H, W
            mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            var = x.data.var(axis=(0, 2, 3), keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)

        output = x_norm * self.weight.data + self.bias.data
        return Tensor(output, requires_grad=x.requires_grad or self.weight.requires_grad)


class LayerNorm(Module):
    """Layer Normalization."""

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


class Embedding(Module):
    """Embedding layer for discrete inputs."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embeddings
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.01)

    def forward(self, input):
        # Input should be indices - ensure they are integers
        indices = input.data.astype(np.int32)
        return Tensor(self.weight.data[indices], requires_grad=self.weight.requires_grad)
