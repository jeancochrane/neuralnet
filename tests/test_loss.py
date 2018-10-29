import unittest

import numpy as np

import env  # Make sure that imports work
import neuralnet.loss as loss
import neuralnet.activation as activation


class TestLoss(unittest.TestCase):

    def test_mse(self):

        mse = loss.MSE(activation.Sigmoid)

        assert mse(np.array([5, 5, 5]), np.array([1, 1, 1])) == 16
        assert mse(5, 5) == 0

        assert mse.delta(100, 5, 1) == 0
        np.testing.assert_array_almost_equal(mse.delta(np.array([1, 1, 1]), np.array([5, 5, 5]), np.array([1, 3, 5])),
                                             np.array([0.786448, 0.393224, 0]))

    def test_log_loss(self):

        log_loss = loss.LogLoss(activation.Sigmoid)

        np.testing.assert_array_almost_equal(log_loss(0.999999, 1), 0)
        assert log_loss(np.exp(-0.5), 1) == 0.5
        assert log_loss(np.array([np.exp(-0.5), np.exp(-0.5)]), np.array([1, 1])) == 0.5

        np.testing.assert_array_equal(log_loss.delta(np.array([1, 1]), np.array([0.5, 0.5]), np.array([1, 1])),
                                      np.array([-0.5, -0.5]))
