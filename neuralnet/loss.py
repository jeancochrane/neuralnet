import numpy as np


def mean_squared_error(X, Y):
    """
    Quadratic error.
    """
    return np.mean((X-Y)**2)
