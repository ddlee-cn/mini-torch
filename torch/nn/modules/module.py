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

    def __getattr__(self, name):
        # bind get attributes method
        pass

    def __setattr__(self, name, value):
        # bind set attributes method: parameter, module and buffer
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, module):
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # del attr
        pass

    def state_dict(self, destination=None, prefix=''):
        # Returns a dictionary containing a whole state of the module.
        for name, param in self._parameters.items():
                destination[prefix + name] = param.data
        for name, buf in self._buffers.items():
                destination[prefix + name] = buf
        for name, module in self._modules.items():
                module.state_dict(destination, prefix + name + '.')
        return destination

    def load_state_dict(self, state_dict):
        # copies parameters and buffers from state_dict into module
        # and its descendants
        own_state = self.state_dict()
        for name, param in state_dict.items():
            param = param.data
            own_state[name].copy_(param)

    def parameters(self, memo=None):
        # return an iterator over module params
        for name, param in self.name_parameters():
            yield param

    def named_parameters(self, memo=None, prefix=''):
        for name, p in self._parameters.items():
            if p is not none and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_childer():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def children(self):
        # Returns an iterator over immediate children modules.
        for name, module in self.named_children():
            yield module

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        # Returns an iterator over all modules in the network.
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self, mode=True):
        # set module into training mode
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        # set module into evaluation mode
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
                p.grad.detach_()

    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers
        return sorted(keys)
