import torch
from torch.nn.parameter import Parameter

from .module import Module
from .. import functional as F


class Threshold(Module):
    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input):
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + str(self.threshold) \
            + ', ' + str(self.value) \
            + inplace_str + ')'


class ReLU(Threshold):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'
