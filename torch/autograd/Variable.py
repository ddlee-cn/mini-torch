import toch._C as _C
from collections import OrderedDict
import torch.sparse as sparse
import torch.utils.hooks as hooks
from ._functions import *
from torch._C import _ImperativeEngine as ImperativeEngine # for backward
Variable._execution_engine = ImperativeEngine()

class Variable(_C._VariableBase):
    # wrapper for  tensor object, holds gradient and a
    # reference to a function that created it
    _fallthrough_methods = {
        'size',
        'stride',
        'nelement',
        'ndimension',
        'element_size',
        'is_contiguous',
        'is_set_to',
        'is_signed',
        'numel',
        'dim',
        'get_device',
        'is_cuda',
    }

    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        raise AttributeError(name)

    def __getitem__(self, key):
        if isinstance(key, Variable) and type(key.data).__name__ == 'ByteTensor':
            return MaskedSelect.apply(self, key)
        return Index.apply(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, Variable) and type(key.data).__name__ == 'ByteTensor':
            if isinstance(value, Variable):
                return MaskedCopy.apply(self, key, value, True)
            else:
                return MaskedFill.apply(self, key, value, True)
        else:
            return SetItem.apply(self, key, value)

    def __deepcopy__(self, memo):
        result = type(self)(self.data.clone())
        result.requires_grad = self.requires_grad
        result.volatile = self.volatile
        memo[id(self)] = result
        return result

    def backward(self, gradient=None, retain_variables=False):
        # computes the gradient of current variable w.r.t graph leaves
        if self.volatile:
            raise RuntimeError
        if gradient is None and self.requires_grad:
            gradient = self.data.new().resize_as_(self.data).fill_(1)
        if not isinstance(gradient, Variable):
            gradient = Variable(gradient, volatile=True)
        self._execution_engine.run_backward((self,), (gradient,), retain_variables)

    def register_hook(self, hook):
        #            hook(grad) -> Variable or None
        if self.volatile:
            raise RuntimeError
        if not self.requires_grad:
            raise RuntimeError
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        # registers a reward obtained as a result of a stochastic process
        self.grad_fn._reinforce(reward)

    def detach(self):
        # Returns a new Variable, detached from the current graph.
        result = NoGrad()(self)
        result._grad_fn = None
        return result

    def detach_(self):
        """Detaches the Variable from the graph that created it, making it a leaf."""
        self._grad_fn = None
        self.requires_grad = False

    @staticmethod
    def _static_blas(cls, args, inplace):
        num_args = len(args)
        alpha = beta = 1
        if num_args > 5:
            raise RuntimeError("too many args")
        if num_args == 5:
            alpha, beta = args[1:3]
        if num_args == 4:
            alpha = args[1]
        return cls.apply(*(args[:1] + args[-2:] + (alpha, beta, inplace)))

    def _blas(self, cls, args, inplace):
        return self._static_blas(cls, (self,) + args, inplace)

    def addmm_(self, *args):
        return self._blas(Addmm, args, True)
