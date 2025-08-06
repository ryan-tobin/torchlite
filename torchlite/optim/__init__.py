"""
Optimization algorithms for TorchLite.
"""

from .optimizer import Optimizer

from .sgd import SGD
from .adam import Adam

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
        LRScheduler,
        StepLR,
        ExponentialLR,
        CosineAnnealingLR,
        ReduceLROnPlateau,
        OneCycleLR,
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
