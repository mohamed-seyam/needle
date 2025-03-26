import numpy as np
import needle as ndl

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(low, high, size=shape))

def logsumexp_forward(shape, axes):
    x = get_tensor(*shape)
    return (ndl.ops.logsumexp(x, axes=axes)).cached_data

def test_op_logsumexp_forward_1():
    np.testing.assert_allclose(
        logsumexp_forward((3, 3, 3), (1, 2)),
        np.array([5.366029, 4.9753823, 6.208126], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_2():
    np.testing.assert_allclose(
        logsumexp_forward((3, 3, 3), None),
        np.array([6.7517853], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_3():
    np.testing.assert_allclose(
        logsumexp_forward((1, 2, 3, 4), (0, 2)),
        np.array(
            [
                [5.276974, 5.047317, 3.778802, 5.0103745],
                [5.087831, 4.391712, 5.025037, 2.0214698],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_4():
    np.testing.assert_allclose(
        logsumexp_forward((3, 10), (1,)),
        np.array([5.705309, 5.976375, 5.696459], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_op_logsumexp_forward_5():
    test_data = ndl.ops.logsumexp(
        ndl.Tensor(np.array([[1e10, 1e9, 1e8, -10], [1e-10, 1e9, 1e8, -10]])), (0,)
    ).numpy()
    np.testing.assert_allclose(
        test_data,
        np.array([1.00000000e10, 1.00000000e09, 1.00000001e08, -9.30685282e00]),
        rtol=1e-5,
        atol=1e-5,
    )
