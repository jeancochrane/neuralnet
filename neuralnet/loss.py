import types

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
        if isinstance(y, (int, float)):
            n = 1
            if not isinstance(a, (int, float)):
                raise TypeError('Param types for a and y must match: ',
                                'a is {a}, y is {y}'.format(a=type(a), y=type(y)))

        elif isinstance(y, (np.ndarray, list, tuple, types.GeneratorType)):
            if isinstance(y, (types.GeneratorType)):
                n = len(list(y))
            else:
                n = len(y)
            if not isinstance(a, (np.ndarray, list, tuple, types.GeneratorType)):
                raise TypeError('Param types for a and y must match: ',
                                'a is {a}, y is {y}'.format(a=type(a), y=type(y)))

        else:
            raise TypeError('y param must be of type int, float, list, tuple, or ndarray,' +
                            ' not {tp}'.format(tp=type(y)))

        return (a - y) / n


class LogLoss(Loss):
    """
    Log (cross-entropy) loss.
    """
    def __call__(self, a, y):
        return -np.mean(np.nan_to_num((y*np.log(a)) + ((1-y)*np.log(1-a))))

    def nabla_C(self, a, y):
        return a - y
