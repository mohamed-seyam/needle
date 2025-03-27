"""The module."""
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
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device = device, dtype = dtype, requires_grad = True))
        self.bias = Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1, device = device, dtype = dtype, requires_grad = True).transpose()) if bias else None
        
    def forward(self, X: Tensor) -> Tensor:
        out = X.matmul(self.weight)
        if self.bias:
            out += self.bias.broadcast_to(out.shape)  # currently needle doesn't support implicit broadcasting
    
        return out
        


class Flatten(Module):
    def forward(self, X):
        raise NotImplementedError()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
        


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for module in self.modules:
            out = module(out)
        return out 
            


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        assert len(logits.shape) == 2 and len(y.shape) == 1
        assert logits.shape[0] == y.shape[0]

        n, k  = logits.shape[0], logits.shape[1] # num of examples, num of classes
        log_sum_exp =  ops.logsumexp(logits, axes=(1,)) # shape (n, )
        y_one_hot = init.one_hot(k, y, device = logits.device, dtype = logits.dtype) # one hot encoding of y (n, k)
        softmax : Tensor = log_sum_exp - (logits * y_one_hot).sum(axes = (1,)) # shape (n, )
        return softmax.sum() / n 


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()