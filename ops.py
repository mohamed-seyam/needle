"""Operator implementations."""

from ast import Index
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy 

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return array_api.add(a,b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


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
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs ,rhs = node.inputs
        return (out_grad * self.scalar * (rhs**(self.scalar -1)), )
        # ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs ,rhs = node.inputs
        
        return (out_grad / rhs , mul_scalar(out_grad , -1) * lhs/(power_scalar(rhs,2)))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes == None :  # when it is None swap the last two axis
          axis1 ,axis2= len(a.shape)-1,  len(a.shape)-2
        else:
          axis1,  axis2 = self.axes[0], self.axes[1]
      
        return array_api.swapaxes(a , axis1 = axis1, axis2 =  axis2)
      
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        print(out_grad.shape)
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.shape == None : 
          return a 
        else:
          return array_api.reshape(a, self.shape) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad.reshape(lhs.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        indices  = []
        node_shape = node.inputs[0].shape
        grad_shape = out_grad.shape

        if len(node_shape) == len(grad_shape):
          for i in range(len(node_shape)):
            if (node_shape[i] < grad_shape[i]):
              indices.append(i)
          return out_grad.sum(tuple(indices)).reshape(node_shape)
        else:
        
          if len(node_shape) !=0:
            for i in range(len(node_shape)):
              if (node_shape[i] < grad_shape[i]):
                indices.append(i)
      
          for k in range(len(node_shape),len(grad_shape)):
            print(k)
            indices.append(k)
          return out_grad.sum(tuple(indices)).reshape(node_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,axis = self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes == None :
          print("i entered in none case")
          return out_grad.broadcast_to(node.inputs[0].shape)

        
        temp_grad = list(out_grad.shape)
        for axis in self.axes:
          temp_grad.insert(axis , 1)

        new_shape = tuple(temp_grad)

        # print(out_grad.shape)
        # print(node.inputs[0].shape)
        new_grad = out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        # print(new_grad)
        return new_grad
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        mat = array_api.matmul(a,b)

        return mat
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        if len(lhs.shape) > len(rhs.shape):
          axis = tuple(i for i in range(len(lhs.shape)-len(rhs.shape)))
          return (out_grad @ rhs.transpose(), lhs.transpose().sum(axes = axis)@ out_grad)
        if len(rhs.shape) > len(lhs.shape):
          axis = tuple(i for i in range(len(rhs.shape)-len(lhs.shape)))
          left_derivitive = (out_grad @ rhs.transpose()).sum(axes = axis)
          print(left_derivitive.shape)
          shape = (*rhs.shape[0:-len(rhs.shape)], *lhs.transpose().shape)
          print(shape)
          right_derivitive = lhs.transpose().broadcast_to(shape) @ out_grad
          print(left_derivitive.shape )
          return left_derivitive, right_derivitive
        return (out_grad @ array_api.transpose(rhs) , array_api.transpose(lhs) @ out_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.__neg__()
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if a.all() > 0:
          return array_api.log(a)
        else:
          print("the value to log is not greater than 0", a)
          raise AssertionError
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.__truediv__(node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(type(node.inputs[0].realize_cached_data()))
        print(node.inputs[0].realize_cached_data().shape)
        data = node.inputs[0].realize_cached_data()

        m,n = node.inputs[0].realize_cached_data().shape
        for i in range(m):
          for k in range(n):
            if data[i][k] > 0 :
              data[i][k] = 1 
            else :
              data[i][k] = 0
          
        
        return out_grad*data

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

