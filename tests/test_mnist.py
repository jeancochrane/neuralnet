import unittest

import numpy as np

import env  # Make sure that imports work


class MNISTTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Import MNIST data.
        """
        # Function to turn batches of nx28x28 px training images into nx784 matrices.
        reshape = lambda df: df.reshape((df.shape[0], 784))

        # Load in the data.
        with np.load('mnist.npz') as mnist:
            x_train, x_test = reshape(mnist['x_train']), reshape(mnist['x_test'])
            y_train, y_test = mnist['y_train'], mnist['y_test']

        # Divide training data into training/validation sets.
        train_idx = np.random.choice(range(x_train.shape[0]), int(0.7*x_train.shape[0]))
        val_idx = np.array([i for i in range(x_train.shape[0]) if i not in train_idx])

        cls.x_train, cls.y_train = x_train[train_idx], y_train[train_idx]
        cls.x_val, cls.y_val = x_train[val_idx], y_train[val_idx]

    def test_setup(self):
        pass
