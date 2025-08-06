import numpy as np
import pytest

from torchlite import Tensor


class TestAutograd:
    """Test automatic differentiation."""

    def test_simple_backward(self):
        """Test backward pass on simple operations."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x + y
        z.backward()

        assert x.grad == 1.0
        assert y.grad == 1.0

    def test_multiplication_backward(self):
        """Test backward pass on multiplication."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()

        assert x.grad == 3.0
        assert y.grad == 2.0

    def test_chain_rule(self):
        """Test chain rule in backward pass."""
        x = Tensor([2.0], requires_grad=True)
        y = x * 3
        z = y + 2
        w = z * z
        w.backward()

        # dw/dx = dw/dz * dz/dy * dy/dx = 2z * 1 * 3 = 2(3x+2) * 3 = 2(8) * 3 =
        # 48
        assert np.isclose(x.grad, 48.0)

    def test_relu_backward(self):
        """Test ReLU backward pass."""
        x = Tensor([-1, 0, 1, 2], requires_grad=True)
        y = x.relu()
        y.sum().backward()

        expected_grad = [0, 0, 1, 1]  # ReLU gradient
        assert np.allclose(x.grad, expected_grad)

    def test_matmul_backward(self):
        """Test matrix multiplication backward pass."""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = Tensor([[5, 6], [7, 8]], requires_grad=True)
        z = x @ y
        z.sum().backward()

        # Gradient of x should be sum of y's columns
        assert np.allclose(x.grad, [[11, 15], [11, 15]])
        # Gradient of y should be sum of x's rows
        assert np.allclose(y.grad, [[4, 4], [6, 6]])

    def test_no_grad(self):
        """Test operations without gradient tracking."""
        x = Tensor([1, 2, 3], requires_grad=False)
        y = x * 2
        assert y.requires_grad == False
        assert y.grad is None

    def test_zero_grad(self):
        """Test gradient zeroing."""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = x * 2
        y.sum().backward()
        assert x.grad is not None

        x.zero_grad()
        assert x.grad is None
