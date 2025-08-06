# Fixed torchlite/nn/__init__.py
"""
Neural network modules for TorchLite.
"""

from .activations import GELU, LeakyReLU, ReLU, Sigmoid, SiLU, Softmax, Tanh
from .attention import MultiheadAttention
from .container import ModuleDict, ModuleList, Sequential
from .layers import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Embedding, LayerNorm, Linear
from .loss import BCELoss, CrossEntropyLoss, L1Loss, MSELoss
from .module import Module, Parameter
from .pooling import AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from .rnn import GRU, LSTM, RNN

__all__ = [
    # Core
    "Module",
    "Parameter",
    # Containers
    "Sequential",
    "ModuleList",
    "ModuleDict",
    # Layers
    "Linear",
    "Conv2d",
    "Dropout",
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "Embedding",
    # Activations
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LeakyReLU",
    "GELU",
    "SiLU",
    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "L1Loss",
    # Pooling
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    # RNN
    "RNN",
    "LSTM",
    "GRU",
    # Attention
    "MultiheadAttention",
]
