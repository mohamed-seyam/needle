from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        raise NotImplementedError()

    def gradient(self, out_grad, node):
        raise NotImplementedError()


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # Step 1: Compute the max along the given axes, keeping dimensions for broadcasting
        max_z_original = array_api.max(Z, axis=self.axes, keepdims=True)

        # Step 2: Subtract max value from Z for numerical stability, then compute the exponential
        exp_shifted = array_api.exp(Z - max_z_original)

        # Step 3: Sum the exponential along the given axes
        sum_exp = array_api.sum(exp_shifted, axis=self.axes)

        # Step 4: Compute the logarithm of the sum of exponential
        log_sum_exp = array_api.log(sum_exp)

        # Step 5: Add the max value back (without keepdims) to restore the correct scale
        max_z_reduce = array_api.max(Z, axis=self.axes)
        result = log_sum_exp + max_z_reduce

        return result
    
    def gradient(self, out_grad, node):
        Z : Tensor = node.inputs[0]
        
        Z_max = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        Z_exp = exp(Z - Z_max)

        Z_sum_exp = Z_exp.sum(axes=self.axes)
        Z_sum_exp = Z_sum_exp.reshape(Z_max.shape)
        Z_sum_exp = broadcast_to(Z_sum_exp, Z_exp.shape)

        out_grad = out_grad.reshape(Z_max.shape)
        out_grad = broadcast_to(out_grad, Z_exp.shape)

        return out_grad * Z_exp / Z_sum_exp
        
    
def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)