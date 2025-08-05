"""Simple data parallel implementation"""
from typing import List
import numpy as np 
from ..nn.module import Module 
from ..tensor import Tensor 

class DaraParallel(Module):
    """
    Simple data parallel wrapper for multi-GPU training.
    This is a simplified version for educational purposes.
    """

    def __init__(self, module: Module, device_ids: List[int] = None):
        super().__init__()
        self.module = module 
        self.device_ids = device_ids or [0]

    def forward(self, *inputs, **kwargs):
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        
        inputs_per_device = self._scatter(inputs, self.device_ids)

        outputs = []
        for i, device_inputs in enumerate(inputs_per_device):
            device_output = self.module(*device_inputs, **kwargs)
            outputs.append(device_output)

        return self._gather(outputs)
    
    def _scatter(self, inputs, device_ids):
        """Split inputs across devices"""
        batch_size = inputs[0].shape[0]
        chunk_size = batch_size // len(device_ids)

        scattered = []
        for i in range(len(device_ids)):
            start = i * chunk_size
            end = start + chunk_size if i < len(device_ids) - 1 else batch_size

            device_inputs = []
            for inp in inputs:
                device_inputs.append(Tensor(inp.data[start:end]))
            scattered.append(tuple(device_inputs))

        return scattered
    
    def _gather(self, outputs):
        """Gather outputs from all devices"""
        gathered_data = np.concatenate([out.data for out in outputs], axis=0)
        return Tensor(gathered_data, requires_grad=outputs[0].requires_grad)
