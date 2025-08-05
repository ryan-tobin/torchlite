"""
TorchLite - A lightweight deep learning framework
"""

__version__ = "0.1.0"

from .tensor import Tensor 
from . import nn 
from . import optim 
from . import data 
from . import utils 

from .nn import Module, Parameter
from .optim import SGD, Adam 

__all__ = [
    "Tensor",
    "nn",
    "optim",
    "data",
    "utils",
    "Module",
    "Parameter",
    "SGD",
    "Adam",
]