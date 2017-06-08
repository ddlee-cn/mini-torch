import torch
from copy import deepcopy
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
                group.setdefault(name, default)

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        packed_state = {(id(k) if isinstance(k, Variable) else k): v
                                    for k,v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}
        state = {id_map.get(k, k): v for k, v in state_dict['state'].items()}

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.zero_()
                    param.grad.detach_()

    def step(self, closure):
        raise NotImplementedError

