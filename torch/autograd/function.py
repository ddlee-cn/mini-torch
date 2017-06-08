import torch

class _ContextMethodMixin(object):

    def save_for_backward(self, *tensors):
        self.to_save = tensors

class FunctionMeta(type):

    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            if 'forward' in super_cls.__dict__:
                has_static_forward = isinstance(super_cls.__dict__['forward'], staticmethod)
                break

        setattr(cls, '_is_legacy', not has_static_forward)

        # old-style functions
        if not has_static_forward:
            return super(FunctionMeta, cls).__init__(name, bases, attrs)

        backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
        setattr(cls, '_backward_cls', backward_fn)

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(with_metaclass(FunctionMeta, _ContextMethodMixin)):
    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(*grad_outputs):
        raise NotImplementedError

class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace
