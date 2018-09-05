import importlib.util
import sys

import signal
from contextlib import contextmanager
import random
import numpy as np

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def sigmoid(x):
    """sigmoid
    Sigmoid function. Output = 1 / (1 + exp(-1)).
    :param x: input
    """

    x = 1/(1+np.exp(-x))
    return x


def sigmoid_grad(a):
    """sigmoid_grad
    Compute gradient of sigmoid.
    :param a: output of the sigmoid function
    """
    
    return (a)*(1-a)


def reLU(x):
    """reLU
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    
    return np.maximum(0,x)


def reLU_grad(a):
    """reLU_grad
    Compute gradient of ReLU.
    :param x: output of ReLU
    """

    grad = np.copy(a)
    grad[grad <= 0] = 0
    grad[grad > 0] = 1
    return grad


def tanh(x):
    """tanh
    Tanh function.
    :param x: input
    """
   
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def tanh_grad(a):
    """tanh_grad
    Compute gradient for tanh.
    :param a: output of tanh
    """

    return 1 - a**2


def softmax(x):
    """softmax
    Softmax function.
    :param x: input
    """

    exp_scores = np.exp(x)
    probs = exp_scores/np.sum(exp_scores, axis = 1, keepdims = True)
    return probs


def softmax_minus_max(x):
    """softmax_minus_max
    Stable softmax function.
    :param x: input
    """

    exp_scores = np.exp(x - np.max(x, axis = 1, keepdims = True))
    probs = exp_scores/np.sum(exp_scores, axis = 1, keepdims = True)
    return probs

score = 0
message = ""

def evaluate_activation(submitted_module, activation_name):
    global message, score
    current_score = 0
    for i in range(10):
        x = np.random.rand(50, 50)
        if not hasattr(submitted_module, activation_name):
            message += 'submitted source has no method {}\n'.format(activation_name)
            return

        submitted_output = getattr(submitted_module, activation_name)(x)
        correct_output = globals()[activation_name](x)
        if type(submitted_output) != type(correct_output):
            message += '{} returns incorrect type\n'.format(activation_name)
        if submitted_output.shape != correct_output.shape:
            message += '{} returns incorrect shape\n'.format(activation_name)
        current_score += np.allclose(submitted_output, correct_output)
    score += current_score
    message += '{} scores: {}\n'.format(activation_name, current_score)

activation_function_names = ['sigmoid', 'sigmoid_grad', 'reLU', 'reLU_grad', 'tanh', 'tanh_grad', 'softmax', 'softmax_minus_max']

try:
    spec = importlib.util.spec_from_file_location("module.name", sys.argv[1])
    submitted_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submitted_module)

    for activation_function_name in activation_function_names:
        evaluate_activation(submitted_module, activation_function_name)

    print(score)
    print(message)
except Exception as e:
    print(score)
    print(e)