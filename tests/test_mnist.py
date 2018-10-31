import unittest

import numpy as np

import env  # Make sure that imports work
import neuralnet as nn


class MNISTTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Import MNIST data.
        """
        # Function to turn batches of nx28x28 px training images into nx784 matrices.
        reshape_x = lambda df: df.reshape((df.shape[0], 784))

        def encode_y(y):
            """
            Function to one-hot encode targets.
            h/t https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
            """
            y_encoded = np.zeros((y.shape[0], 10))
            y_encoded[np.arange(y.shape[0]), y] = 1
            return y_encoded

        # Load in the data.
        with np.load('mnist.npz') as mnist:
            x_train, x_test = reshape_x(mnist['x_train']), reshape_x(mnist['x_test'])
            y_train, y_test = encode_y(mnist['y_train']), encode_y(mnist['y_test'])

        # Divide training data into training/validation sets.
        train_idx = np.random.choice(range(x_train.shape[0]), int(0.7*x_train.shape[0]))
        val_idx = np.array([i for i in range(x_train.shape[0]) if i not in train_idx])

        cls.x_train, cls.y_train = x_train[train_idx], y_train[train_idx]
        cls.x_val, cls.y_val = x_train[val_idx], y_train[val_idx]

    def test_evaluate(self):
        net = nn.network.Network((784, 30, 10))

        net.train(self.x_train, self.y_train)

        metrics = net.evaluate(self.x_val, self.y_val)

        # 1 should be a more-than-reasonable upper bound for the mean loss
        self.assertTrue(metrics['mean_loss'] < 1)
