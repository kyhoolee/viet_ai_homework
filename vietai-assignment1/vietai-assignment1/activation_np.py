"""activation_np.py
This file provides activation functions for the NN 
Author: Phuong Hoang
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    TODO: 
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    #[TODO 1.1]
    result = 1.0 / (1.0 + np.exp(-x))
    return result


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    result = sigmoid(x) * (1.0 - sigmoid(x))
    return None


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    result = np.max(0, x)
    return result


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    
    grad = a.copy()
    grad[grad<=0] = 0
    grad[grad>0] = 1
    return grad


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    result = (1 - np.exp(-2.0 * x)) / (1 + np.exp(-2.0 * x))
    return result


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    result = 1.0 - np.tanh(x)**2
    return result


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """
    output = np.exp(x)
    output = output / np.sum(output)
    return output


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """
    output = np.exp(w - np.max(w))
    output = e / np.sum(e)
    return None
