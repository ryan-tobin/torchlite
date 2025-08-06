"""
TorchLite - A lightweight deep learning framework
"""

__version__ = "0.1.0"

from . import nn, optim
from .tensor import Tensor

try:
    from . import data
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import data module: {e}. Some functionality may be limited.", ImportWarning
    )
    data = None

try:
    from . import utils
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import utils module: {e}. Some functionality may be limited.", ImportWarning
    )
    utils = None

from .nn import Module, Parameter
from .optim import SGD, Adam

__all__ = [
    "Tensor",
    "nn",
    "optim",
    "Module",
    "Parameter",
    "SGD",
    "Adam",
]

if data is not None:
    __all__.append("data")
if utils is not None:
    __all__.append("utils")

try:
    import numpy as np

    _np_version = tuple(map(int, np.__version__.split(".")[:2]))
    if _np_version < (1, 19):
        warnings.warn(
            f"TorchLite requires NumPy >= 1.19.0, but found {np.__version__}. "
            "Some features may not work correctly.",
            RuntimeWarning,
        )
except ImportError:
    raise ImportError("TorchLite requires NumPy. Please install it with: pip install numpy>=1.19.0")


def _get_version_info():
    """Get version information for debugging."""
    info = {
        "torchlite": __version__,
        "numpy": np.__version__,
    }

    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = "Not installed"

    try:
        import PIL

        info["PIL"] = PIL.__version__
    except ImportError:
        info["PIL"] = "Not installed"

    return info


version_info = _get_version_info
