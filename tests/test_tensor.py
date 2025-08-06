import numpy as np
import pytest

from torchlite import Tensor


class TestTensorBasics:
    """Test basic tensor operations"""

    def test_tensor_creation(self):
        """Test tensor creation from different inputs."""
        # From list
        t1 = Tensor([1, 2, 3])
        assert t1.shape == (3,)

        # From numpy array
        arr = np.array([[1, 2], [3, 4]])
        t2 = Tensor(arr)
        assert t2.shape == (2, 2)

        # From scalar
        t3 = Tensor(5.0)
        assert t3.shape == ()

    def test_tensor_attributes(self):
        """Test tensor attributes"""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.requires_grad
        assert t.grad is None
        assert t.dtype == np.float32

    def test_tensor_arithmetic(self):
        """Test basic arithmetic operations"""
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])

        # Add
        z = x + y
        assert np.allclose(z.data, [5, 7, 9])

        # Multiply
        z = x * y
        assert np.allclose(z.data, [4, 10, 18])

        # Scalar Ops
        z = x + 10
        assert np.allclose(z.data, [11, 12, 13])

    def test_tensor_matmul(self):
        """Test matrix multiplication"""
        x = Tensor([[1, 2], [3, 4]])
        y = Tensor([[5, 6], [7, 8]])
        z = x @ y
        expected = [[19, 22], [43, 50]]
        assert np.allclose(z.data, expected)

    def test_tensor_reduction(self):
        """Test reduction operations"""
        x = Tensor([[1, 2, 3], [4, 5, 6]])

        # Sum
        assert np.allclose(x.sum().data, 21)
        assert np.allclose(x.sum(axis=0).data, [5, 7, 9])
        assert np.allclose(x.sum(axis=1).data, [6, 15])

        # Mean
        assert np.allclose(x.mean().data, 3.5)
        assert np.allclose(x.mean(axis=0).data, [2.5, 3.5, 4.5])
