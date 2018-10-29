import types

import numpy as np

from . import activation


class Loss(object):
    """
    Abstract base class for a loss function.
    """
    def __call__(self, a, y):
        """
        The loss function itself.

        :param z: Weighted input for a layer.

        :return:  The activation corresponding to the weighted input.
        """
        raise NotImplementedError('Loss functions must define the __call__ method.')

    def delta(self, z, a, y):
        """
        The output "error," dC/dz, of the last layer in the network. Used to compute the
        first step in backpropagation.

        :param z: Weighted inputs to a layer.
        :param a: Output activations for a layer.
        :param y: Expected labels for a layer.

        :return: Output error of the last layer in the network, dC/dz.
        """
        raise NotImplementedError('Loss functions must define the delta method.')


class MSE(Loss):
    """
    Quadratic loss.
    """
    def __init__(self, activation):
        self.activation = activation

    def __call__(self, a, y):
        return np.mean((a-y)**2)

    def delta(self, z, a, y):
        activation = self.activation()
        return (a - y) * activation.ddz(z)


class LogLoss(Loss):
    """
    Log (cross-entropy) loss.
    """
    def __init__(self, activation):
        self.activation = activation

    def __call__(self, a, y):
        return -np.mean(np.nan_to_num((y*np.log(a)) + ((1-y)*np.log(1-a))))

    def delta(self, z, a, y):
        return a - y
