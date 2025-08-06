"""
Optimization algorithms for TorchLite.
"""

from .adam import Adam
from .optimizer import Optimizer
from .sgd import SGD

try:
    from .rmsprop import RMSprop
except ImportError:
    RMSprop = None

try:
    from .adagrad import Adagrad
except ImportError:
    Adagrad = None

try:
    from .lr_scheduler import (
        CosineAnnealingLR,
        ExponentialLR,
        LRScheduler,
        OneCycleLR,
        ReduceLROnPlateau,
        StepLR,
    )

    _schedulers_available = True
except ImportError:
    _schedulers_available = False
    LRScheduler = None
    StepLR = None
    ExponentialLR = None
    CosineAnnealingLR = None
    ReduceLROnPlateau = None
    OneCycleLR = None

__all__ = ["Optimizer", "SGD", "Adam"]

if RMSprop is not None:
    __all__.append("RMSprop")

if Adagrad is not None:
    __all__.append("Adagrad")

if _schedulers_available:
    __all__.extend(
        [
            "LRScheduler",
            "StepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
            "OneCycleLR",
        ]
    )
