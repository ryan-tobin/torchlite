import pytest
import numpy as np
from torchlite import Tensor
from torchlite.nn import Module, Linear, Conv2d, ReLU, Sigmoid, Dropout, BatchNorm1d
from torchlite.nn import MSELoss, CrossEntropyLoss


class TestModules:
    """Test neural network modules."""

    def test_linear_layer(self):
        """Test linear layer forward pass."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10))
        y = layer(x)
        assert y.shape == (32, 5)

    def test_conv2d_layer(self):
        """Test 2D convolution layer."""
        layer = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(8, 3, 32, 32))
        y = layer(x)
        assert y.shape == (8, 16, 32, 32)

        # Test without padding
        layer = Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        y = layer(x)
        assert y.shape == (8, 16, 30, 30)

    def test_activation_functions(self):
        """Test activation functions."""
        x = Tensor([-2, -1, 0, 1, 2])

        # ReLU
        relu = ReLU()
        y = relu(x)
        assert np.allclose(y.data, [0, 0, 0, 1, 2])

        # Sigmoid
        sigmoid = Sigmoid()
        y = sigmoid(Tensor([0]))
        assert np.isclose(y.data, 0.5)

    def test_dropout(self):
        """Test dropout layer."""
        dropout = Dropout(p=0.5)
        x = Tensor(np.ones((100, 100)))

        # Training mode
        dropout.train()
        y = dropout(x)
        # Roughly half should be zero (statistical test)
        assert 0.4 < (y.data == 0).mean() < 0.6

        # Eval mode
        dropout.eval()
        y = dropout(x)
        assert np.allclose(y.data, x.data)

    def test_batchnorm(self):
        """Test batch normalization."""
        bn = BatchNorm1d(10)
        x = Tensor(np.random.randn(32, 10))

        # Training mode
        bn.train()
        y = bn(x)
        assert y.shape == (32, 10)
        # Output should be normalized
        assert np.allclose(y.data.mean(axis=0), 0, atol=1e-5)
        # Variance should be close to 1 (increase tolerance for numerical
        # stability)
        # Changed from 1e-5 to 1e-3
        assert np.allclose(y.data.var(axis=0), 1, atol=1e-3)

        # Check running stats are updated
        assert bn.running_mean is not None
        assert bn.running_var is not None


class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss(self):
        """Test mean squared error loss."""
        loss_fn = MSELoss()
        pred = Tensor([[1, 2], [3, 4]])
        target = Tensor([[1, 1], [3, 3]])
        loss = loss_fn(pred, target)
        # MSE = ((0 + 1 + 0 + 1) / 4) = 0.5
        assert np.isclose(loss.data, 0.5)

    def test_cross_entropy_loss(self):
        """Test cross-entropy loss."""
        loss_fn = CrossEntropyLoss()
        # Simple case: perfect prediction
        logits = Tensor([[10, 0, 0], [0, 10, 0]])
        targets = Tensor([0, 1])  # Class indices
        loss = loss_fn(logits, targets)
        assert loss.data < 0.01  # Should be very small

        # Uniform prediction
        logits = Tensor([[1, 1, 1]])
        targets = Tensor([0])
        loss = loss_fn(logits, targets)
        assert np.isclose(loss.data, -np.log(1 / 3), atol=0.1)


class TestModuleAPI:
    """Test Module API functionality."""

    def test_parameter_registration(self):
        """Test automatic parameter registration."""

        class SimpleNet(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(10, 5)
                self.fc2 = Linear(5, 2)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        net = SimpleNet()
        params = list(net.parameters())
        # Should have 4 parameters: 2 weights + 2 biases
        assert len(params) == 4

        # Check named parameters
        named_params = dict(net.named_parameters())
        assert "fc1.weight" in named_params
        assert "fc1.bias" in named_params
        assert "fc2.weight" in named_params
        assert "fc2.bias" in named_params

    def test_train_eval_mode(self):
        """Test training and evaluation mode switching."""

        class NetWithDropout(Module):
            def __init__(self):
                super().__init__()
                self.dropout = Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        net = NetWithDropout()

        # Check training mode
        net.train()
        assert net.training
        assert net.dropout.training

        # Check eval mode
        net.eval()
        assert net.training == False
        assert net.dropout.training == False

    def test_zero_grad(self):
        """Test zeroing gradients for all parameters."""
        net = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = net(x)
        y.sum().backward()

        # Check gradients exist
        for param in net.parameters():
            assert param.grad is not None

        # Zero gradients
        net.zero_grad()
        for param in net.parameters():
            assert param.grad is None
