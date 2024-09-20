import sys 
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.simple_ml import *

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_parse_mnist():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])


def test_softmax_loss_ndl():
    # test forward pass for log
    np.testing.assert_allclose(
        ndl.log(ndl.Tensor([[4.0], [4.55]])).numpy(),
        np.array([[1.38629436112], [1.515127232963]]),
    )

    # test backward pass for log
    gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4)))

    X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    np.random.seed(0)
    Z = ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    np.testing.assert_allclose(
        softmax_loss(Z, y).numpy(), 2.3025850, rtol=1e-6, atol=1e-6
    )
    Z = ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
    np.testing.assert_allclose(
        softmax_loss(Z, y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6
    )

    # test softmax loss backward
    Zsmall = ndl.Tensor(np.random.randn(16, 10).astype(np.float32))
    ysmall = ndl.Tensor(y_one_hot[:16])
    gradient_check(softmax_loss, Zsmall, ysmall, tol=0.01, backward=True)