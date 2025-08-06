from collections import OrderedDict
from .module import Module


class Sequential(Module):
    """Sequential container of modules."""

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ModuleList(Module):
    """List of modules."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module):
        self._modules[str(len(self._modules))] = module

    def extend(self, modules):
        for module in modules:
            self.append(module)

    def __getitem__(self, idx):
        return self._modules[str(idx)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList doesn't have a forward method")


class ModuleDict(Module):
    """Dictionary of modules."""

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[key] = module

    def update(self, modules):
        if isinstance(modules, dict):
            for key, module in modules.items():
                self[key] = module
        else:
            for key, module in modules:
                self[key] = module

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict doesn't have a forward method")
