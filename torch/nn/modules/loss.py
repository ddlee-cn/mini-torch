from torch.autograd import Variable
import torch
from .module import Module
from .. import functional as F

class _Loss(Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class MSELoss(_Loss):
    pass
    # already got in getattr(self._backend, type(self).__name__)
