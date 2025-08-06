"""Model summary and visualization utils."""

from typing import List, Tuple, Dict
import numpy as np
from ..nn.module import Module
from collections import OrderedDict
from ..tensor import Tensor


def summary(model: Module, input_shape: Tuple[int, ...], batch_size: int = -1):
    """
    Print a summary of the model architecture.
    Similar to Keras model.summary()
    """

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary_list)

            m_key = f"{class_name}-{module_idx}"
            summary_list[m_key] = OrderedDict()
            summary_list[m_key]["input_shape"] = list(input[0].shape)
            summary_list[m_key]["output_shape"] = list(output.shape)

            params = 0
            for param in module.parameters():
                params += np.prod(param.shape)

            summary_list[m_key]["nb_params"] = params

        if not hasattr(module, "_forward_hooks"):
            module._forward_hooks = OrderedDict()

        module._forward_hooks[len(module._forward_hooks)] = hook

    summary_list = OrderedDict()
    hooks = []

    model.apply(register_hook)

    model.eval()
    x = Tensor(np.random.randn(2, *input_shape[1:]))
    model(x)

    for h in hooks:
        h.remove()

    print("-" * 64)
    print(f"{'Layer (type)':<20} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 64)

    total_params = 0
    trainable_params = 0

    for layer in summary_list:
        print(
            f"{layer:<20} {str(summary_list[layer]['output_shape']):<25}"
            f"{summary_list[layer]['nb_params']:<15}"
        )
        total_params += summary_list[layer]["nb_params"]
        trainable_params += summary_list[layer]["nb_params"]

    print("=" * 64)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: 0")
    print("-" * 64)


def get_model_size(model: Module) -> Dict[str, int]:
    """
    Get model size statistics.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with size information
    """
    total_params = 0
    total_size_mb = 0

    for name, param in model.named_parameters():
        n_params = np.prod(param.shape)
        size_mb = n_params * 4 / (1024 * 1024)
        total_params += n_params
        total_size_mb += size_mb

    return {
        "total_parameters": total_params,
        "trainable_parameters": total_params,
        "non_trainable_parameters": 0,
        "model_size_mb": total_size_mb,
    }


def count_parameters(model: Module) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: Model to count parameters for

    Returns:
        Total parameter count
    """
    return sum(np.prod(p.shape) for p in model.parameters())
