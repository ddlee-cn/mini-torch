import torch
from . import _functions

def linear(input, weight, bias=None):
    if bieas is None:
        return _functions.linear.Linear.apply(input, weight)
    else:
        return _functions.linear.Linear.apply(input, weight, bias)

def threshold(input, threshold, value, inplace=False):
    return _functions.thnn.auto.Threshold(threshold, value, inplace)(input)

def relu(input, inplace=False):
    return _functions.thnn.auto.Threshold(0, 0, inplace)(input)
