import numpy as np


class Activation(object):
    """
    Abstract base class for an activation function.
    """
    def __call__(self, z):
        """
        The activation function itself.

        :param z: Weighted input for a layer.

        :return:  The activation corresponding to the weighted input.
        """
        raise NotImplementedError('Activation functions must define the __call__ method.')

    def ddz(self, z):
        """
        The derivative of the activation function with respect to the weighted
        inputs to a layer.

        :param z: Weighted input for a layer.

        :return:  Derivative of a layer's activation.
        """
        raise NotImplementedError('Activation functions must define the ddz method.')


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def ddz(self, z):
        return self.__call__(z) * (1 - self.__call__(z))


class Linear(Activation):
    """
    Linear activation function.
    """
    def __call__(self, z):
        return z

    def ddz(self, z):
        return 1
