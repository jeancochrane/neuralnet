import numpy as np


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

    def nabla_C(self, a, y):
        """
        The vector of partial derivatives of the cost function with respect
        to the activations in a layer.

        :param a: Output activations for a layer.
        :param y: Expected labels for a layer.

        :return:  Partial derivatives of a layer's cost with respect to the activations.
        """
        raise NotImplementedError('Loss functions must define the nabla_C method.')


class MSE(Loss):
    """
    Quadratic loss.
    """
    def __call__(self, a, y):
        return np.mean((a-y)**2)

    def nabla_C(self, a, y):
        return (a - y) / len(y)
