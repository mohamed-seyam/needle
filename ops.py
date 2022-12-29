"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later work
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

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
        return a** self.scalar
      
    def gradient(self, out_grad, node):
    
        val  = node.inputs[0]
        return (out_grad * self.scalar * (power_scalar(val, self.scalar -1)), )
    


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        
        return array_api.divide(a,b)
        
    def gradient(self, out_grad, node):
        
        lhs ,rhs = node.inputs
        return (out_grad / rhs , mul_scalar(out_grad , -1) * lhs/(power_scalar(rhs,2)))
        


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
      
        if self.axes == None : 
          axis1 ,axis2= len(a.shape)-1,  len(a.shape)-2
        else:
          axis1,  axis2 = self.axes[0], self.axes[1]
      
        return array_api.swapaxes(a , axis1 = axis1, axis2 =  axis2)


    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)
     


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if self.shape == None  or a.shape == (): 
          return a 
        else:
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
       
        return array_api.sum(a,axis = self.axes)
      

    def gradient(self, out_grad, node):
      
        if self.axes == None :
          return out_grad.broadcast_to(node.inputs[0].shape)

        
        temp_grad = list(out_grad.shape)
        if type (self.axes) == int :   ## for only one int axis 
          temp_grad.insert(self.axes, 1)
        else:
          for axis in self.axes:
            temp_grad.insert(axis , 1)

        new_shape = tuple(temp_grad)

        new_grad = out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
 
        return new_grad
=


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
       
        mat = array_api.matmul(a,b)
        return mat

    def gradient(self, out_grad, node):
        
        lhs, rhs = node.inputs
       
        if len(lhs.shape) > len(rhs.shape):
         
          axis = tuple(i for i in range(len(lhs.shape)-len(rhs.shape)))
          return (out_grad @ rhs.transpose(), lhs.transpose().sum(axes = axis)@ out_grad)
        
        if len(rhs.shape) > len(lhs.shape):
         
          axis = tuple(i for i in range(len(rhs.shape)-len(lhs.shape)))
          left_derivitive = (out_grad @ rhs.transpose()).sum(axes = axis)
          shape = (*rhs.shape[0:-len(rhs.shape)], *lhs.transpose().shape)
          right_derivitive = lhs.transpose().broadcast_to(shape) @ out_grad
          return left_derivitive, right_derivitive
       
        return (out_grad @ array_api.transpose(rhs) , array_api.transpose(lhs) @ out_grad)
       

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
      
        return array_api.negative(a)
    

    def gradient(self, out_grad, node):
  
        return out_grad.__neg__()


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
      
        if a.all() > 0:
          return array_api.log(a)
        
        else:
          
          raise AssertionError

    def gradient(self, out_grad, node):
        return out_grad.__truediv__(node.inputs[0])
     


def log(a):
    return Log()(a)


class Max(TensorOp):
    def __init__(self, axes : Optional[tuple] = None):
      self.axes = axes

    def compute(self, a):
      return array_api.max(a, axis = self.axes)

    def gradient(self, out_grad, node):
      raise NotImplementedError()

def max(a, axes = None):
  return Max(axes)(a)



class Exp(TensorOp):
    def compute(self, a):
        
        return array_api.exp(a)
   

    def gradient(self, out_grad, node):
        
        return out_grad * array_api.exp(node.inputs[0])
       


def exp(a):
    return Exp()(a)



class ReLU(TensorOp):
    def compute(self, a):
    
        return array_api.maximum(0,a)
  

    def gradient(self, out_grad, node):
     
        # old solution it also work
        # data = node.inputs[0].realize_cached_data()
        # m,n = node.inputs[0].realize_cached_data().shape
        # for i in range(m):
        #   for k in range(n):
        #     if data[i][k] > 0 :
        #       data[i][k] = 1 
        #     else :
        #       data[i][k] = 0
        # return out_grad * data
        dividend = relu(node.inputs[0])
        divisor = add_scalar(dividend, numpy.finfo(float).eps)
        mask = dividend / divisor
        return mask * out_grad
                
       


def relu(a):
    return ReLU()(a)




class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        z_shape = list(Z.shape)
        if self.axes is not None :
          for ax in self.axes:
            z_shape[ax] = 1

        
          max_val = array_api.reshape(array_api.max(Z, axis = self.axes), tuple(z_shape))
          diff = Z - max_val
          exp_diff = array_api.exp(diff)
          sum_exp_diff = array_api.sum(exp_diff , axis = self.axes)
          log_sum_exp_diff = array_api.log(sum_exp_diff)
          result = log_sum_exp_diff + array_api.reshape(max_val, log_sum_exp_diff.shape)
          print(result)
        else:
          print("i entered here")
          result = array_api.log(array_api.sum(array_api.exp(Z-array_api.max(Z)))) + array_api.max(Z)
        return result
    
    def gradient(self, out_grad, node):

        z = node.inputs[0]
      
        print("input axes :", self.axes)
        print("shape of z : ", z.shape)
        print("gradient shape is ", out_grad.shape)


        if self.axes is not None : 
         
          temp_grad_shape = list(out_grad.shape)
         
          for ax in self.axes :
            temp_grad_shape.insert(ax ,1)
         
          out_grad = out_grad.reshape(tuple(temp_grad_shape))
          print(out_grad.shape)
          

          max_val = z.max(axes = self.axes).reshape(out_grad.shape)
          numerator = exp(z - max_val)
          print("numerator shape is :", numerator.shape)
         
          dominator = exp(z-max_val).sum(axes = self.axes).reshape(out_grad.shape)
          print("dominator shape is :", dominator.shape)
          
    
          
        else :

          max_val = z.max()
          numerator = exp(z-max_val)
          dominator = exp(z-max_val).sum()
      
       
        result =  out_grad * (numerator / dominator)
        print("result shape is ", result.shape)


        return result


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

