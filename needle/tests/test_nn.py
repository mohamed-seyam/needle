import numpy as np 
import needle as ndl
import needle.nn as nn

def nn_linear_bias_init():
    np.random.seed(1337)
    f = ndl.nn.Linear(7, 4)
    return f.bias.cached_data


def nn_linear_weight_init():
    np.random.seed(1337)
    f = ndl.nn.Linear(7, 4)
    f.weight.cached_data
    return f.weight.cached_data



def test_nn_linear_weight_init_1():
    np.testing.assert_allclose(
        nn_linear_weight_init(),
        np.array(
            [
                [-4.4064468e-01, -6.3199449e-01, -4.1082984e-01, -7.5330488e-02],
                [-3.3144259e-01, 3.4056887e-02, -4.4079605e-01, 8.8153863e-01],
                [4.3108878e-01, -7.1237373e-01, -2.1057765e-01, 2.3793796e-01],
                [-6.9425780e-01, 8.9535803e-01, -1.0512712e-01, 5.3615785e-01],
                [5.4460180e-01, -2.5689366e-01, -1.5534532e-01, 1.5601574e-01],
                [4.8174453e-01, -5.7806653e-01, -3.9223823e-01, 3.1518409e-01],
                [-6.5129338e-04, -5.9517515e-01, -1.6083106e-01, -5.5698222e-01],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_bias_init_1():
    np.testing.assert_allclose(
        nn_linear_bias_init(),
        np.array([[0.077647, 0.814139, -0.770975, 1.120297]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return ndl.Tensor(np.random.randint(low, high, size=shape))

def linear_forward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = ndl.nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1])
    x = get_tensor(*rhs_shape)
    return f(x).cached_data


def linear_backward(lhs_shape, rhs_shape):
    np.random.seed(199)
    f = ndl.nn.Linear(*lhs_shape)
    f.bias.data = get_tensor(lhs_shape[-1]) # manually set the bias - lhs_shape[-1] => out_feature
    x = get_tensor(*rhs_shape)  # rhs_shape => (num_of_examples, input_feature)
    (f(x) ** 2).sum().backward()  # compute the loss 
    return x.grad.cached_data # return the gradient

def relu_forward(*shape):
    f = ndl.nn.ReLU()
    x = get_tensor(*shape)
    return f(x).cached_data


def relu_backward(*shape):
    f = ndl.nn.ReLU()
    x = get_tensor(*shape)
    (f(x) ** 2).sum().backward()
    return x.grad.cached_data

def sequential_forward(batches=3):
    np.random.seed(42)
    f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
    x = get_tensor(batches, 5)
    return f(x).cached_data


def sequential_backward(batches=3):
    np.random.seed(42)
    f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
    x = get_tensor(batches, 5)
    f(x).sum().backward()
    return x.grad.cached_data


def test_nn_linear_forward_1():
    np.testing.assert_allclose(
        linear_forward((10, 5), (1, 10)),
        np.array([[3.849948, 9.50499, 2.38029, 5.572587, 5.668391]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_forward_2():
    np.testing.assert_allclose(
        linear_forward((10, 5), (3, 10)),
        np.array(
            [
                [7.763089, 10.086785, 0.380316, 6.242502, 6.944664],
                [2.548275, 7.747925, 5.343155, 2.065694, 9.871243],
                [2.871696, 7.466332, 4.236925, 2.461897, 8.209476],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_forward_3():
    np.testing.assert_allclose(
        linear_forward((10, 5), (1, 3, 10)),
        np.array(
            [
                [
                    [4.351459, 8.782808, 3.935711, 3.03171, 8.014219],
                    [5.214458, 8.728788, 2.376814, 5.672185, 4.974319],
                    [1.343204, 8.639378, 2.604359, -0.282955, 9.864498],
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_backward_1():
    np.testing.assert_allclose(
        linear_backward((10, 5), (1, 10)),
        np.array(
            [
                [
                    20.61148,
                    6.920893,
                    -1.625556,
                    -13.497676,
                    -6.672813,
                    18.762121,
                    7.286628,
                    8.18535,
                    2.741301,
                    5.723689,
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_backward_2():
    print(linear_backward((10, 5), (3, 10)))
    np.testing.assert_allclose(
        linear_backward((10, 5), (3, 10)),
        np.array(
            [
                [
                    24.548800,
                    8.775347,
                    4.387898,
                    -21.248514,
                    -3.9669373,
                    24.256767,
                    6.3171115,
                    6.029777,
                    0.8809935,
                    3.5995162,
                ],
                [
                    12.233745,
                    -3.792646,
                    -4.1903896,
                    -5.106719,
                    -12.004269,
                    11.967942,
                    11.939469,
                    19.314493,
                    10.631226,
                    14.510731,
                ],
                [
                    12.920014,
                    -1.4545978,
                    -3.0892954,
                    -6.762379,
                    -9.713004,
                    12.523148,
                    9.904757,
                    15.442993,
                    8.044141,
                    11.4106865,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_linear_backward_3():
    print(linear_backward((10, 5), (1, 3, 10)))
    np.testing.assert_allclose(
        linear_backward((10, 5), (1, 3, 10)),
        np.array(
            [
                [
                    [
                        16.318823,
                        0.3890714,
                        -2.3196607,
                        -10.607947,
                        -8.891977,
                        16.04581,
                        9.475689,
                        14.571134,
                        6.581477,
                        10.204643,
                    ],
                    [
                        20.291656,
                        7.48733,
                        1.2581345,
                        -14.285493,
                        -6.0252004,
                        19.621624,
                        4.343303,
                        6.973201,
                        -0.8103489,
                        4.037069,
                    ],
                    [
                        11.332953,
                        -5.698288,
                        -8.815561,
                        -7.673438,
                        -7.6161675,
                        9.361553,
                        17.341637,
                        17.269142,
                        18.1076,
                        14.261493,
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_relu_forward_1():
    np.testing.assert_allclose(
        relu_forward(2, 2),
        np.array([[3.35, 4.2], [0.25, 4.5]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_relu_backward_1():
    np.testing.assert_allclose(
        relu_backward(3, 2),
        np.array([[7.5, 2.7], [0.6, 0.2], [0.3, 6.7]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_sequential_forward_1():
    print(sequential_forward(batches=3))
    np.testing.assert_allclose(
        sequential_forward(batches=3),
        np.array(
            [
                [3.296263, 0.057031, 2.97568, -4.618432, -0.902491],
                [2.465332, -0.228394, 2.069803, -3.772378, -0.238334],
                [3.04427, -0.25623, 3.848721, -6.586399, -0.576819],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )

def test_nn_sequential_backward_1():
    np.testing.assert_allclose(
        sequential_backward(batches=3),
        np.array(
            [
                [0.802697, -1.0971, 0.120842, 0.033051, 0.241105],
                [-0.364489, 0.651385, 0.482428, 0.925252, -1.233545],
                [0.802697, -1.0971, 0.120842, 0.033051, 0.241105],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )

def softmax_loss_forward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = ndl.nn.SoftmaxLoss()
    return np.array(f(x, y).cached_data)

def softmax_loss_backward(rows, classes):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = ndl.nn.SoftmaxLoss()
    loss = f(x, y)
    loss.backward()
    return x.grad.cached_data

def test_nn_softmax_loss_forward_1():
    np.testing.assert_allclose(
        softmax_loss_forward(5, 10),
        np.array(4.041218, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_softmax_loss_forward_2():
    np.testing.assert_allclose(
        softmax_loss_forward(3, 11),
        np.array(3.3196716, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_softmax_loss_backward_1():
    np.testing.assert_allclose(
        softmax_loss_backward(5, 10),
        np.array(
            [
                [
                    0.00068890385,
                    0.0015331834,
                    0.013162163,
                    -0.16422154,
                    0.023983022,
                    0.0050903494,
                    0.00076135644,
                    0.050772052,
                    0.0062173656,
                    0.062013146,
                ],
                [
                    0.012363418,
                    0.02368262,
                    0.11730081,
                    0.001758993,
                    0.004781439,
                    0.0029000894,
                    -0.19815083,
                    0.017544521,
                    0.015874943,
                    0.0019439887,
                ],
                [
                    0.001219767,
                    0.08134181,
                    0.057320606,
                    0.0008595553,
                    0.0030001428,
                    0.0009499555,
                    -0.19633561,
                    0.0008176346,
                    0.0014898272,
                    0.0493363,
                ],
                [
                    -0.19886842,
                    0.08767337,
                    0.017700946,
                    0.026406704,
                    0.0013147127,
                    0.0107361665,
                    0.009714483,
                    0.023893777,
                    0.019562569,
                    0.0018656658,
                ],
                [
                    0.007933789,
                    0.017656967,
                    0.027691642,
                    0.0005605318,
                    0.05576411,
                    0.0013114461,
                    0.06811045,
                    0.011835824,
                    0.0071787895,
                    -0.19804356,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nn_softmax_loss_backward_2():
    np.testing.assert_allclose(
        softmax_loss_backward(3, 11),
        np.array(
            [
                [
                    0.0027466794,
                    0.020295369,
                    0.012940894,
                    0.04748398,
                    0.052477922,
                    0.090957515,
                    0.0028875037,
                    0.012940894,
                    0.040869843,
                    0.04748398,
                    -0.33108455,
                ],
                [
                    0.0063174255,
                    0.001721699,
                    0.09400159,
                    0.0034670753,
                    0.038218185,
                    0.009424488,
                    0.0042346967,
                    0.08090791,
                    -0.29697907,
                    0.0044518122,
                    0.054234188,
                ],
                [
                    0.14326698,
                    0.002624026,
                    0.0032049934,
                    0.01176007,
                    0.045363605,
                    0.0043262867,
                    0.039044812,
                    0.017543964,
                    0.0037236712,
                    -0.3119051,
                    0.04104668,
                ],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )
