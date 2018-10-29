import unittest

import numpy as np

import env  # Make sure that imports work
import neuralnet.activation as activation

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.reshape(60000, 784)