"""Simple JIT tracing for optimization"""
from typing import Callable, Any 
import inspect 
from ..nn.module import Module 

class TracedModule:
    """A traced version of a module for optimization"""

    def __init__(self, module: Module, example_inputs):
        self.module = module 
        self.graph = self._trace(example_inputs)

    def _trace(self, example_inputs):
        """Trace the computation graph."""
        graph = {
            'inputs': example_inputs,
            'operations': [],
            'outputs': None 
        }

        outputs = self.module(example_inputs)
        graph['outputs'] = outputs 

        return graph 
    
    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def optimize(self):
        pass 

def trace(module: Module, example_inputs):
    """Trace a module with example inputs."""
    return TracedModule(module, example_inputs)
