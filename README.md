# mini-torch

This repo serves as a collection of my observations of the DNN Framework [PyTorch](https://pytorch.org).

> Nothing can't be solved by adding a layer of abstraction. If can't, add one more.

## Motivation

At first, I've implemented LSTM using PyTorch library, taking advantage of the `nn.Module` class. Then I decided to explore how exactly the PyTorch library implements the LSTM structure and wrote an [article](https://blog.ddlee.cn/2017/05/29/LSTM-Pytorch%E5%AE%9E%E7%8E%B0/) about that.

I'm impressed with the abstractions the library established. Then it comes this repo.


## The Structure
In my view, there is 3 main abstraction level: Module, Function, Backends.

- Module is the most fundamantal componet of network. It take inputs of Variable, call Functions and Feed forward
- Functions are the heroes behind scene. Many modules just wrap a Function.
- Well, Backends are the real unnamed heroes behind scene. They are mostly implemented in effecient C codes and binded for python.

## Three Essential Class
There are 3 core class for the network to work.

- nn.Module abstracts layers or a stack of layers in a network
- autograd.Variable abstracts the data flow in a network
- autograd.Function abstracts the ops executed in a network

### nn.Module
state_dict:
- Parameters: the Variable updating as traing
- Buffers: the Variable inside a module
- Module: the recurrent structure of nn.Module class
### autograd.Variable
state_dict:
- requires_grad: bool, control whether update the data in Variable
- volatile: bool, for switch between train and test mode
- _backward_hooks: tracker of the lifecycle of Variable

### autograd.Function
methods:
- forward
- backward
