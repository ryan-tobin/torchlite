"""
Utility functions and tools for TorchLite.
"""

from .data import (
    create_data_splits,
    download_url,
    extract_archive,
    get_data_dir,
    split_dataset,
    verify_checksum,
)
from .model_summary import count_parameters, get_model_size, summary
from .serialization import (
    deserialize_model,
    load,
    load_checkpoint,
    save,
    save_checkpoint,
    serialize_model,
)
from .visualization import GradientFlow, plot_loss_curves, visualize_activations

__all__ = [
    # Data utilities
    "download_url",
    "extract_archive",
    "verify_checksum",
    "split_dataset",
    "create_data_splits",
    "get_data_dir",
    # Serialization
    "save",
    "load",
    "save_checkpoint",
    "load_checkpoint",
    "serialize_model",
    "deserialize_model",
    # Model utilities
    "summary",
    "get_model_size",
    "count_parameters",
    # Visualization
    "GradientFlow",
    "plot_loss_curves",
    "visualize_activations",
]
