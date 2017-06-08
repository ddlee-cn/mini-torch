import torch
from . import _functions

def linear(input, weight, bias=None):
    if bieas is None:
        return _functions.linear.Linear.apply(input, weight)
    else:
        return _functions.linear.Linear.apply(input, weight, bias)
