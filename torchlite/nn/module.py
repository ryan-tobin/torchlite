"""Module Implementation"""

from typing import Dict, Iterator, Tuple , Union
from collections import OrderedDict
from ..tensor import Tensor
import numpy as np 

class Parameter(Tensor):
    """A special Tensor that is automatically added to Module's parameters."""
    
    def __init__(self, data: Union[Tensor, np.ndarray, list]):
        if isinstance(data, Tensor):
            super().__init__(data.data, requires_grad=True)
        else:
            super().__init__(data, requires_grad=True)

class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self.training = True 

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value 
        elif isinstance(value, Module):
            self._modules[name] = value 
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def parameters(self) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        for name, param in self.named_parameters():
            yield param 

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters with their names."""
        for name, param in self._parameters.items():
            yield name, param 
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param 

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        self.training = mode 
        for module in self._modules.values():
            module.train(mode)
        return self 
    
    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def forward(self, *args, **kwargs):
        """Forward pass. Should be overriden by all subclasses."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    