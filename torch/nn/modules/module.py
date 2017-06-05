import torch
from torch.autograd import Variable
from ..parameter import Parameter
import torch.utils.hooks as hooks


class Module(object):
    dump_patches = False

    def __init__(self):
        self._backend = thnn_backend
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def forward(self, *input):
        raise NotImplementedError

    def register_buffer(self, name, tensor):
        # adds a persistent buffer to the module, like runing_mean in BN
        self._buffers[names] = tensor

    def register_parameter(self, name, param):
        # adds a parameter to the module
        self._parameters[name] = param

    def add_module(self, name, module):
        # adds a child module
        self._modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad.data is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        return self

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self, device_id=None):
        # change type or location of model parameters and buffers
        return self._apply(lambda t: t.cunda(device_id))
    # similar ones:
    # def cpu(), type(), float(), double()

    def register_backward_hook(self, hook):
        # hook(module, grad_input, grad_output) -> Tensor or None
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook):
        # hook(module, input, output) -> None
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._foward_hooks[handle.id] = hook
        return handle

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        for hook in self._forward(*input, **kwargs):
            hook_result = hook(self, input, result)
        var = result
        while not isinstance(var, Variable):
            var = var[0]
        grad_fn = var.grad_fn
        if grad_fn is not None and len(self._backward_hooks) > 0:
            for hook in self._backward_hooks.values():
                wrapper = functools.partial(hook, self)
                functools.update_wrapper(wrapper, hook)
                grad_fn.register_hook(wrapper)
        return result
