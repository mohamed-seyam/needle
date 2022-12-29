"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.kaiming_uniform(self.in_features, self.out_features)
        self.bias  = init.kaiming_uniform(fan_in = self.out_features, fan_out = 1).transpose()

            

    def forward(self, X: Tensor) -> Tensor:
        y = X @ self.weight + self.bias.broadcast_to((X.shape[0],self.out_features))
        return y




class Flatten(Module):
    def multiply_list(self, list_arr):
        mult = 1
        for ele in list_arr :
          mult *= ele
        print()
        return mult
   
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape_list = list(X.shape)
        return X.reshape((shape_list.pop(0), self.multiply_list(shape_list)))
        
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        print("i get before")
        modules_list = self._children()  # return child modules
        for module in modules_list:
          x = module(x)
        return x 
        ### END YOUR SOLUTION

import numpy as np

class SoftmaxLoss(Module):

    # lhs is logsumexp(logits)  and rhs is z_y
    def forward(self, logits: Tensor, y: Tensor):
        m, n = logits.shape
        # n = int((y.max()+1).numpy())
        one_hot_y = init.one_hot(n,y)
        print("shape of product is : ", (logits * one_hot_y).shape)
        rhs = (logits * one_hot_y).sum(axes = 1)
        print("rhs shape", rhs.shape)
        # lhs = ops.logsumexp(logits)
        lhs = ops.log(ops.exp(logits).sum(axes = 1))
        print("lhs shape", lhs.shape)

        return (lhs-rhs).sum(axes = 0) / m
       
        



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim)).reshape((1,dim))
        self.bias = Parameter(init.zeros((dim))).reshape((1,dim))
        self.running_mean = init.zeros((dim))
        self.running_var = init.ones((dim))
 


    def forward(self, x: Tensor) -> Tensor:

        if self.training:
          mean = (x.sum(axes = 0) / x.shape[0])
          mean_shaped = mean.reshape((1,x.shape[1])).broadcast_to(x.shape)
      
          var = (((x- mean_shaped)**2).sum(axes = 0) / x.shape[0])
          var_shaped = var.reshape((1,x.shape[1])).broadcast_to(x.shape)
       
          self.running_mean = mean.detach() * self.momentum + (1-self.momentum) * self.running_mean
          self.running_var= self.momentum * var.detach() + (1-self.momentum) * self.running_var
          
          numerator = x - mean_shaped
          denominator = (var_shaped + self.eps)**.5

        ## during testing you want to use the self.running statistics that actually learnend 
        else : 

          running_mean_shaped = self.running_mean.reshape((1,x.shape[1])).broadcast_to(x.shape)
          running_var_shaped = self.running_var.reshape((1,x.shape[1])).broadcast_to(x.shape)

          numerator = x - running_mean_shaped
          denominator = (running_var_shaped + self.eps)**.5


        return (self.weight.broadcast_to(x.shape) * (numerator / denominator)) + self.bias.broadcast_to(x.shape)



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
     
        self.weight = Parameter(
          init.ones(dim, device = device , dtype = dtype).reshape((1,dim)))
        
        self.bias = Parameter(
          init.zeros(dim, device = device, dtype = dtype).reshape((1,dim)))
     

    def forward(self, x: Tensor) -> Tensor:
        
        n = x.shape[1]
        mean = (x.sum(axes = 1 ) / n).reshape((x.shape[0],1)).broadcast_to(x.shape)
        var = ((x-mean)**2).sum(axes = 1) / n
        std = ((var + self.eps) ** .5).reshape((x.shape[0],1)).broadcast_to(x.shape)
        y = (x-mean)/ std 
        result = self.weight * y + self.bias
        return result





class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training : 

          binary = init.randb(*x.shape, p = 1-self.p)

          return (x * binary)/ (1-self.p) 



class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:

        raise NotImplementedError()




