"""
Neural network modules for TorchLite.
"""

from .module import Module, Parameter
from .layers import Linear, Conv2d, Dropout, BatchNorm1d, BatchNorm2d, LayerNorm
from .activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU, SiLU
from .loss import MSELoss, CrossEntropyLoss, BCELoss, L1Loss
from .pooling import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
from .rnn import RNN, LSTM, GRU
from .attention import MultiheadAttention
from .container import Sequential, ModuleList, ModuleDict
from .embedding import Embedding

__all__ = [
    # Core
    'Module',
    'Parameter',
    
    # Layers
    'Linear',
    'Conv2d',
    'Dropout',
    'BatchNorm1d',
    'BatchNorm2d',
    'LayerNorm',
    
    # Activations
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'LeakyReLU',
    'GELU',
    'SiLU',
    
    # Loss functions
    'MSELoss',
    'CrossEntropyLoss',
    'BCELoss',
    'L1Loss',
    
    # Pooling
    'MaxPool2d',
    'AvgPool2d',
    'AdaptiveAvgPool2d',
    
    # RNN
    'RNN',
    'LSTM', 
    'GRU',
    
    # Attention
    'MultiheadAttention',
    
    # Containers
    'Sequential',
    'ModuleList',
    'ModuleDict',

    # Embedding
    'Embedding'
]