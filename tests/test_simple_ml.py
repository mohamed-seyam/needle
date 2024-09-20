import sys 
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.simple_ml import *

def test_parse_mnist():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
    

def test_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)

    Z = np.zeros((y.shape[0], 10))
    n_values = np.max(y) + 1
    y_one_hot = np.eye(n_values)[y]

    np.testing.assert_allclose(softmax_loss(ndl.Tensor(Z), ndl.Tensor(y_one_hot)).numpy(), 2.3025850)
    Z = np.random.randn(y.shape[0], 10)
    np.testing.assert_allclose(softmax_loss(ndl.Tensor(Z), ndl.Tensor(y_one_hot)).numpy(), 2.7291998)

def test_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    
    np.random.seed(0)

    Z = np.zeros((y.shape[0], 10))
    n_values = np.max(y) + 1
    y_one_hot = np.eye(n_values)[y]

    np.testing.assert_allclose(softmax_loss(ndl.Tensor(Z), ndl.Tensor(y_one_hot)).numpy(), 2.3025850)
    Z = np.random.randn(y.shape[0], 10)
    np.testing.assert_allclose(softmax_loss(ndl.Tensor(Z), ndl.Tensor(y_one_hot)).numpy(), 2.7291998)
