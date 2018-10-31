import unittest

import numpy as np

import env  # Make sure that imports work
from neuralnet.network import Network


class TestNetwork(unittest.TestCase):

    def test_untrained_predict(self):
        net = Network((4, 5, 1))

        predicted = net.predict([1, 1, 1, 1])
        expected = np.array([0.5])

        self.assertEqual(predicted.shape, (1,))
        self.assertTrue(predicted < 1)

    def test_initialized_weight_dimensions(self):
        net = Network((4, 5, 1))

        weight_shape = [w.shape for w in net.weights]

        self.assertEqual(weight_shape, [(4, 5), (5, 1)])

    def test_initialized_bias_dimensions(self):
        net = Network((4, 5, 1))

        bias_shape = [b.shape for b in net.biases]

        self.assertEqual(bias_shape, [(5,), (1,)])

    def test_train(self):
        net = Network((4, 5, 1))
        net.train([np.array([1, 2, 3, 4])], [np.array([1.])])
