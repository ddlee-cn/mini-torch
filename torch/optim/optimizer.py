import torch
from collections import defalutdict
from torch.autograd import Variable

class Optimizer(object):
    def __init__(self, params, defaults):
        self.state = defaultdict(dict)
        self.param_groups = list(params)

        param_set = set()
        for group in self.param_groups:
            group['params'] = list(group['params'])
            group_set = set(group['params'])
            param_set.update(group_set)

        for name, default in defaults.items():
            for i, group in enumerate(self.param_groups):

