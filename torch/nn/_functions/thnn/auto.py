# auto import functions from C backend
# in the example case Threshold and MSELoss
import torch
from torch.autograd.function import Function
from . import _all_functions

def _generate_function_classes(scope_dict):
    global function_list, function_by_name
    function_list = parse_header(THNN_H_PATH) # function implemented in _C.THNN.h
    function_by_name = {fn.name: fn for fn in function_list}
    classes_to_generate = {fn.name.partition('_')[0] for fn in function_list}
    exceptions = {} # function implemented in python
    name_remap = {
        'MSECriterion': 'MSELoss',
        }
    classes_to_generate -= exceptions
    for fn in classes_to_generate:
        update_output = function_by_name[fn + '_updateOutput']
        update_grad_input = function_by_name[fn + '_updateGradInput']
        acc_grad_parameters = function_by_name.get(fn + '_accGradParameters')
        class_name = name_remap.get(fn, fn)
        # This has to call a function to retain correct references to functions
        if 'Criterion' in fn:
            cls = _make_function_class_criterion(class_name, update_output,
                                                 update_grad_input, acc_grad_parameters)
        else:
            cls = _make_function_class(class_name, update_output,
                                       update_grad_input, acc_grad_parameters)
        scope_dict[class_name] = cls
        if not class_name.startswith('_'):
            _all_functions.append(cls)
_generate_function_classes(locals())
