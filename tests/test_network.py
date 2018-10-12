import unittest

import numpy as np

import env  # Make sure that imports work
from neuralnet.network import Network


class TestNetwork(unittest.TestCase):

    def test_untrained_predict(self):
        net = Network((4, 5, 1))

        predicted = net.predict([1, 1, 1, 1])
        expected = np.array([2])

        np.testing.assert_array_equal(predicted, expected)
