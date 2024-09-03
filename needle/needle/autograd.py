import needle 
from needle import init
from needle import ops
from .backend_numpy import Device, cpu, all_devices
import numpy as array_api  # we will use it this way because we will change it in the future for our implementation
from typing import Tuple, Union, Optional, List, Dict
import numpy

TENSOR_COUNTER = 0


LAZY_MODE = False
"""in the context of deep learning frameworks there are generallyIn the context of deep learning frameworks: 

Eager Execution and Graph (or Deferred) Execution.

1. **Eager Execution**: In this mode, operations are executed immediately as they are called from Python. 
This provides more interactivity and easier debugging. TensorFlow 2.0 and PyTorch use eager execution by default. 
This mode is more Pythonic and allows you to use Python's debugging tools. However, it can be less efficient because it doesn't allow for optimizations like operation fusion.

2. **Graph (or Deferred) Execution (lazy mode)**: In this mode, operations are not executed immediately. 
Instead, a computational graph is built, and then the graph is executed all at once. 
This mode allows for optimizations like operation fusion, parallelism, and deployment on different devices (like GPUs). 
TensorFlow 1.x used graph execution by default. However, this mode can be harder to debug because errors are not raised until the graph is executed, 
which might be far from where the error was coded.
"""

NDArray = numpy.ndarray





class Op:
    """Operator definition"""

    def __call__(self, *args):
        raise NotImplementedError()
    
    def compute(self, *args: Tuple[NDArray]):
        """Calculate the forward path of operator.

        Parameters
        ----------
        input: NDArray
            A list of input arrays to the function 

        Returns
        -------
        output: NDArray
            Array output of the operation
        
        """

        raise NotImplementedError()
    
    def gradient(
            self, out_grad: "Value", node: "Value"
    )-> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint of the output of the operation

        node : Value
            The value node of forward evaluation

        Returns
        -------
        input_grads: Value, Tuple[Value]
            A list containing partial adjoint to be propagated to each of the input node.
        
        """
        raise NotImplementedError()
    
    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        
        elif isinstance(output, list):
            return tuple(output)
        
        else:
            return (output,)
        

class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structure"""
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    
class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)

class Value:

    """Value that represents a node in a computational graph. 
    Each Value instance can be the result of an operation (Op) applied to some inputs, 
    or it can be a leaf node (an input to the computational graph that's not the result of any operation).

    Attributes:
        op (Optional[Op]): The operation that produced this value. If this is None, it means this value is not the result of any operation (i.e., it's a leaf node).
        inputs (List[Value]): A list of the inputs to the operation. If this value is a leaf node, this list is empty.
        cached_data (NDArray): The result of the operation. This is None until the operation is computed.
        requires_grad (bool): A boolean indicating whether this value requires a gradient. This is used for automatic differentiation.
    """
    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """This method computes the operation and caches the result. If the result is already cached, it returns the cached result."""
        # avoid recomputation 
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data, in other words, it will return value of each input
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for  x in self.inputs]
        )
        return self.cached_data
    
    def is_leaf(self):
        """This method checks whether this value is a leaf node (i.e., not the result of any operation)..

        Returns:
            bool: True if the value is a leaf node, False otherwise.
        """
        return self.op is None

    def __del__(self):
        """This method is called when the instance is about to be destroyed. It decrements a global counter TENSOR_COUNTER"""
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
            self,
            op: Optional[Op],
            inputs: List["Tensor"],
            *,
            num_outputs: int = 1,
            cached_data: List[object] = None,
            requires_grad: Optional[bool] = None
    ):
        """This method is similar to __init__, but it's not automatically called when an instance is created. Instead, 
        it's manually called in make_const to initialize an instance. It increments TENSOR_COUNTER, 
        determines whether this value requires a gradient, and sets the op, inputs, cached_data, and requires_grad attributes.
        and it used here rather than __init__ because we want to control the creation of the object, sometimes we want to create a constant value
        and other time we want to create a value from an operation. so different way of initialization is needed."""
        
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1 
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad
    
    @classmethod
    def make_const(cls, data, *, requires_grad: False):
        """Create a constant value.

        Args:
            data: The data of the constant value.
            requires_grad (bool, optional): Whether the constant value requires gradient. Defaults to False.

        Returns:
            Value: The created constant value.
        """
        value = cls.__new__(cls)
        value._init(
            None, 
            [],
            cached_data = data,
            requires_grad = requires_grad
        )
        return value
    

    @classmethod
    def make_from_op(cls, op: Op, inputs = List["Value"]):
        value  = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad: 
                return value.detach()
            
            value.realize_cached_data()
        
        return value
    

class TensorTuple(Value):
    """Value that represents a tuple of tensors. This is used to represent the output of an operation that returns multiple tensors."""
    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)
    
    #! look at this in future
    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)
        
    def tuple(self):
        """the self object is expected to be iterable and return a tuple"""
        return tuple([x for x in self])
    
    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())
    
    def __str__(self):
        return self.__repr__()
    
    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])
    
    def detach(self):
        """Create a new Tensor that share the same data but detaches from the graph"""
        return Tuple.make_const(self.realize_cached_data())
        

class Tensor(Value):
    """Tensor class that represents a tensor in a computational graph. This class is a subclass of Value.
       Note: it inherits from Value class, so it has all the attributes and methods of Value class.
    """
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,   # the * means that the following arguments are keyword-only
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):  # as the array is a tensor so you expect it has the methods we will implement 
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod # 
    def _array_from_numpy(numpy_array, device, dtype):
        """Convert a numpy array to a tensor array. This method is used to convert a numpy array to a tensor array when creating a new tensor.
        currently we define the array_api as numpy so we will use it as numpy.array(numpy_array, dtype=dtype) but in the future we will change it to our implementation
        and in the future implementation it should have some api like it should have its method similar to numpy.array
        """
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property  
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    # the following dunder methods are used to overload the operators
    def __add__(self, other):
        """Overload the + operator."""
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        """Overload the * operator."""
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        """Overload the ** operator."""
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        """Overload the - operator. when binary operator is used. for example x - y it will overload by this way x.__sub__(y)"""
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        """Overload the / operator."""
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        """Overload the @ operator."""
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        """Overload the - operator. when unary operator is used. for example -x. it will overload by this way x.__neg__()"""
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__
    
    """
    These lines are defining the "right" versions of the arithmetic operations for the class. 
    In Python, when you perform an operation involving two objects, like a + b, Python first tries to call a.__add__(b). 
    If that doesn't work (for example, if a doesn't have an __add__ method or if a.__add__(b) returns NotImplemented), 
    then Python tries to call b.__radd__(a).

    The "r" in __radd__, __rmul__, etc. stands for "right". 
    These methods are called when the object is on the right side of the operation and the left object doesn't support the operation.

    In previous code, __radd__ = __add__, __rmul__ = __mul__, etc. are saying that the right versions of the operations should behave 
    the same as the left versions. This is common for types where the operations are commutative (i.e., the order of the operands doesn't matter), like numbers.
    For example, with these definitions, if a is an instance of this class and b is a number, both a + b and b + a will do the same thing: they will call a.__add__(b). 
    If a.__add__(b) returns NotImplemented (which means a doesn't know how to add a number), then Python will call b.__radd__(a), which is the same as a.__add__(b), 
    so it will try the addition again.
    """


def compute_gradient_of_variables(output_tensor, out_grad):
    """
    Computes the gradient of the `output_tensor` with respect to each node in the computational graph.

    This function performs reverse-mode automatic differentiation, commonly known as backpropagation, 
    to compute the gradients of a given output node with respect to all the variables (nodes) in the 
    computational graph. The gradients are stored in the `grad` attribute of each node (Variable).

    Parameters:
    -----------
    output_tensor : Tensor
        The output node of the computational graph for which the gradient is to be computed.
        
    out_grad : Tensor
        The gradient of the loss function with respect to the `output_tensor`. 
        This is typically set to 1 when calculating gradients for scalar loss functions.

    Returns:
    --------
    None
        The gradients are computed and stored in-place in the `grad` attribute of each node.

    Detailed Explanation:
    ---------------------
    1. The function initializes a dictionary (`node_to_output_grads_list`) that maps each node to 
       a list of gradient contributions from each of its output nodes.
       
    2. It performs a reverse topological sort of the computational graph starting from the `output_tensor`.
       
    3. It then iterates over the nodes in reverse topological order, computing the gradient for each node.
       - If the node is an operation node (i.e., it has a non-None `op`), it calls the `gradient_as_tuple` 
         method to calculate the gradient with respect to its inputs.
       - The computed gradients are then propagated backward through the graph, updating the `grad` attribute 
         of each node.
       
    4. For each node, the gradient contributions from all paths are summed up using `sum_node_list`, and 
       this final gradient is stored in the node's `grad` attribute.

    Notes:
    ------
    - The function assumes that the computational graph is acyclic and that the `find_topo_sort` function 
      returns a valid topological ordering.
      
    - This function is designed to work with scalar loss functions. If you are working with vector-valued 
      outputs, you may need to adjust the `out_grad` accordingly.

    Example Usage:
    --------------
    ```python
    # Assuming Tensor, Variable, and the relevant operations are defined

    # Create a simple computational graph
    a = Variable(np.array(2.0), name="a")
    b = Variable(np.array(3.0), name="b")
    c = a * b  # Multiplication operation
    d = c + b  # Addition operation
    output = d.sum()  # Sum operation to produce a scalar output

    a (2.0) ----
                \
                * ---- c = a * b (6.0) ----
                /                            \
    b (3.0) ----                              + ---- d = c + b (9.0) ---- sum() ---- output (9.0)



    # Initialize the gradient of the loss with respect to the output (usually 1)
    out_grad = np.array(1.0)

    # Compute gradients with respect to each node
    compute_gradient_of_variables(output, out_grad)

    # Access gradients
    print("Gradient of a:", a.grad)  # Should print the gradient of the output with respect to 'a'
    print("Gradient of b:", b.grad)  # Should print the gradient of the output with respect to 'b'
    print("Gradient of c:", c.grad)  # Should print the gradient of the output with respect to 'c'
    print("Gradient of d:", d.grad)  # Should print the gradient of the output with respect to 'd'
    ```
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    
    # node_to_output_grad_list
    # {f: [out_grad from loss],
    # d: [partial grad from f],
    # e: [partial grad from f],
    # c: [grad from d, grad from e],
    # a: [partial grad from c],
    # b: [partial grad from c]
    # }

    # outer loop to get gradient of each node 
    # final states of grad field of each node
    # f.grad = out_grad from loss
    # d.grad = partial grad from f
    # e.grad = partial grad from f
    # c.grad = sum (grad from d, grad from e)
    # a.grad = partial grad from c
    # b.grad = partial grad from c

    #inner loop to get gradient of each node with respect to it is inputs vi = vi+1_bar * partial vi+1 partial vi

    print(len(reverse_topo_order))

   
    for node in reverse_topo_order:
      if node_to_output_grads_list.get(node) is not None:
        node.grad = sum_node_list(node_to_output_grads_list[node])
        if node.op is not None:
          input_grads = node.op.gradient_as_tuple(node.grad , node)
          for idx, inp in enumerate(node.inputs):
            if node_to_output_grads_list.get(inp) is None:
              node_to_output_grads_list[inp] = []
              node_to_output_grads_list[inp].append(input_grads[idx])
            else:
              node_to_output_grads_list[inp].append(input_grads[idx])
        else:
          continue
          
      else:
        continue



def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """

    N = len(node_list)
    visited = set()
    topo_order = []
    for node in node_list:
      topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    
    if id(node) in visited:
      return 
    visited.add(id(node))
    for chld in node.inputs:
      topo_sort_dfs(chld, visited, topo_order)
    topo_order.append(node)


##############################
####### Helper Methods #######
##############################

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)

