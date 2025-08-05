"""
CUDA support for TorchLite.
Optional module that requires CuPy installation.

!! NOT AVAILABLE ON macOS !!
"""

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

from .kernels import (
    cuda_add, cuda_multiply, cuda_matmul,
    cuda_relu, cuda_sigmoid, cuda_tanh
)

__all__ = [
    'CUDA_AVAILABLE',
    'cuda_add',
    'cuda_multiply', 
    'cuda_matmul',
    'cuda_relu',
    'cuda_sigmoid',
    'cuda_tanh',
]

def is_available():
    """Check if CUDA is available."""
    return CUDA_AVAILABLE

def device_count():
    """Get number of CUDA devices."""
    if CUDA_AVAILABLE:
        return cp.cuda.runtime.getDeviceCount()
    return 0

def current_device():
    """Get current CUDA device."""
    if CUDA_AVAILABLE:
        return cp.cuda.runtime.getDevice()
    return None

def set_device(device_id):
    """Set current CUDA device."""
    if CUDA_AVAILABLE:
        cp.cuda.runtime.setDevice(device_id)