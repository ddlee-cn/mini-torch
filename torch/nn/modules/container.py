from collections import OrderedDict
import torch
import string
from .module import Module

class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        for key, value in kwargs.items():
            self.add_module(key, value)


    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)


class Sequential(Module):
    def __getitem__(self, idx):
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
