import numpy as np

from .storage import _StorageBase
from .tensor import _TensorBase

class FloatStorage(_C.FloatStorageBase, _StorageBase):
    pass

class FloatTensor(_C.FloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage

from .functional import *

import torch.autograd
import torch.nn
import torch.optim
