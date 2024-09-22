"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

"""
define a helper add() function, to avoid the need to call EWiseAdd()(a,b) (which is a bit cumbersome) to add two Tensor objects. 
"""
def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return (out_grad * self.scalar * (power_scalar(x, self.scalar -1)),)
    


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b 

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs * rhs)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes == None:
            axis1, axis2 = len(a.shape) -1 , len(a.shape) -2
        else:
            axis1, axis2 = self.axes[0], self.axes[1]
        
        return array_api.swapaxes(a, axis1, axis2)
    
    def gradient(self, out_grad, node):
        return out_grad.transpose(axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
    
    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return out_grad.reshape(lhs.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        indices = [] # store the indices of the dimensions that need to be summed
        
        node_shape = node.inputs[0].shape
        grad_shape = self.shape  # or out_grad.shape

        # Pad node_shape with ones on the left to match the length of grad shapes
        len_diff = len(grad_shape) - len(node_shape)
        node_shape_padded = (1,) * len_diff + node_shape

        indices = []

        for i in range(len(grad_shape)):
            if node_shape_padded[i] == 1 and grad_shape[i] > 1:
                indices.append(i)

        if indices:
            out_grad = out_grad.sum(axes = tuple(indices))

        return out_grad.reshape(node_shape)
         

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        node_shape = node.inputs[0].shape # shape of the input before summation
        
        if self.axes is None:
            # If no axes are provided, sum reduces to a scalar, broadcast to the original shape
            return out_grad.broadcast_to(node_shape)
        
        # Normalize axes (handle negative axes)
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        axes = tuple([axis if axis >= 0 else axis + len(node_shape) for axis in axes])

        # Create a shape to broadcast to (inserting 1s where dimensions were reduced)
        grad_shape = list(out_grad.shape)
        for axis in sorted(axes):
            grad_shape.insert(axis, 1)

        new_shape = tuple(grad_shape)
        
        # Reshape and broadcast the gradient to the original input shape
        return out_grad.reshape(new_shape).broadcast_to(node_shape)

        



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a
    

    def gradient(self, out_grad, node):
        return - out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad / a


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
       return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        grad_mask = a.numpy() > 0
        return out_grad * Tensor(grad_mask)


def relu(a):
    return ReLU()(a)