import random

import numpy as np

from . import activation
from . import loss


class Network(object):

    def __init__(self, dims, activation=activation.Sigmoid, cost=loss.MSE):
        # Initialize (zero) weights and biases based on the dimensions.
        self.weights = [np.zeros((dims[i-1], dims[i])) for i in range(1, len(dims))]
        self.biases = [np.zeros(dim) for dim in dims]
        self.biases = self.biases[1:]  # The first layer is the input layer, so it doesn't need a bias

        # Initialize activation and cost functions.
        self.activation = activation()
        self.cost = cost(activation)

    def predict(self, X):
        """
        One forward pass of the network (approximate f(X) given X).
        """
        # Make sure that the input dimensions match the weights.
        assert len(X) == self.weights[0].shape[0], \
        "Input of length {ln} does not match first weight vector with dimensions {dim}".format(
            ln=len(X),
            dim=self.weights[0].shape
        )

        Y = np.array(X)
        for w, b in zip(self.weights, self.biases):
            Y = self.activation(np.dot(Y, w) + b)
        return Y

    def train(self, X, Y, batch_size=1, num_epochs=100, learning_rate=0.01):

        """
        Train the model.
        """
        training_data = list(zip(X, Y))

        for epoch in range(num_epochs):
            # Take a sample of size batch_size from the training data.
            sample = random.sample(training_data, batch_size)

            features = [training[0] for training in sample]
            targets = [training[1] for training in sample]

            # Compute the gradient.
            delta_w, delta_b = self.backprop(features, targets)

            # Update weights and biases based on the gradient.
            self.weights = [w - (learning_rate * dw) for w, dw in zip(self.weights, delta_w)]
            self.biases = [b - (learning_rate * db) for b, db in zip(self.biases, delta_b)]

    def backprop(self, X, Y):
        """
        Calculate the gradient for a network.

        :param features: Training examples.
        :param labels:   Expected output for each feature.

        :return: Tuple of (delta_w, delta_b), the gradients for the weights and biases,
                 respectively.
        """
        # Initialize output layer activation to the input layer.
        aL = X

        # Initialize variables for keeping track of inputs and activations to
        # each layer as we iterate through the network.
        a, z = [aL], []

        # Compute inputs and activations for each layer.
        for wl, bl in zip(self.weights, self.biases):
            zl = np.dot(aL, wl) + bl
            aL = self.activation(zl)

            a.append(aL)
            z.append(zl)

        output_error = self.cost.delta(z[-1], a[-1], Y)

        # Initialize outputs
        delta_w = [np.zeros(wi.shape) for wi in self.weights]
        delta_b = [np.zeros(bi.shape) for bi in self.biases]

        # Set the gradient for the output layer
        delta_b[-1] = output_error  # BP3
        delta_w[-1] = aL * output_error  # BP4

        # Backwards pass
        for i in range(len(self.weights)-1, 0, -1):
            output_error = (np.dot(output_error, np.transpose(self.weights[i])) *
                            self.activation.ddz(z[i-1]))  # BP2

            b_prime = output_error
            w_prime = a[i] * output_error

        return delta_w, delta_b
