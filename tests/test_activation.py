import unittest

import numpy as np

import env  # Make sure that imports work
import neuralnet.activation as activation


class TestActivation(unittest.TestCase):

    def test_sigmoid(self):

        sigmoid = activation.Sigmoid()

        assert sigmoid(0) == 0.5
        assert np.round(sigmoid(-np.log(0.33333333)), 4) == 0.75

        assert sigmoid.ddz(0) == 0.25
        assert sigmoid.ddz(100) == 0

    def test_linear(self):

        linear = activation.Linear()

        assert linear(0) == 0
        assert linear(1) == 1
        assert linear(10) == 10

        assert linear.ddz(0) == 1
        assert linear.ddz(1) == 1
        assert linear.ddz(10) == 1
