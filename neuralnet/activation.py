import numpy as np


def sigmoid(z):
    """
    Sigmoid function (the default activation function for vanilla neural nets.)
    """
    return 1 / 1 + np.exp(-z)
