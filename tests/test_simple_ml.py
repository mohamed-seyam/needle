import sys 
import os
import numdifftools as nd

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


def test_nn_epoch_ndl():
    # test forward/backward pass for relu
    np.testing.assert_allclose(
        ndl.relu(
            ndl.Tensor(
                [
                    [-46.9, -48.8, -45.45, -49.0],
                    [-49.75, -48.75, -45.8, -49.25],
                    [-45.65, -45.25, -49.3, -47.65],
                ]
            )
        ).numpy(),
        np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    gradient_check(ndl.relu, ndl.Tensor(np.random.randn(5, 4)))

    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    W1_0, W2_0 = W1.copy(), W2.copy()
    W1 = ndl.Tensor(W1)
    W2 = ndl.Tensor(W2)
    X_ = ndl.Tensor(X)
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    dW1 = nd.Gradient(
        lambda W1_: softmax_loss(
            ndl.relu(X_ @ ndl.Tensor(W1_).reshape((5, 10))) @ W2, y_
        ).numpy()
    )(W1.numpy())
    dW2 = nd.Gradient(
        lambda W2_: softmax_loss(
            ndl.relu(X_ @ W1) @ ndl.Tensor(W2_).reshape((10, 3)), y_
        ).numpy()
    )(W2.numpy())
    W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(
        dW1.reshape(5, 10), W1_0 - W1.numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        dW2.reshape(10, 3), W2_0 - W2.numpy(), rtol=1e-4, atol=1e-4
    )

    # test full epoch
    X, y = parse_mnist(
        "data/train-images-idx3-ubyte.gz", "data/train-labels-idx1-ubyte.gz"
    )
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    np.testing.assert_allclose(
        np.linalg.norm(W1.numpy()), 28.437788, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(W2.numpy()), 10.455095, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        loss_err(ndl.relu(ndl.Tensor(X) @ W1) @ W2, y),
        (0.19770025, 0.06006667),
        rtol=1e-4,
        atol=1e-4,
    )