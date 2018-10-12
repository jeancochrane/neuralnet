import numpy as np

from . import activation
from . import loss


class Network(object):

    def __init__(self, dims, activation=activation.sigmoid, cost=loss.mean_squared_error):
        # Initialize (zero) weights and biases based on the dimensions.
        self.weights = [np.zeros((dims[i-1], dims[i])) for i in range(1, len(dims))]
        self.biases = [np.zeros(dim) for dim in dims]
        self.biases = self.biases[1:]  # The first layer is the input layer, so it doesn't need a bias

        # Initialize activation and cost functions.
        self.activation = activation
        self.cost = cost

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
        training_data = zip(X, Y)

        for epoch in range(num_epochs):
            # Take a sample of size batch_size from the training data.
            sample = np.random.sample(training_data, size=batch_size, replace=False)

            predictions = self.predict([training[0] for training in sample])

            gradient = self.calculate_gradient(predictions, [pair[1] for pair in sample])

            # Update weights based on the gradient.

            # Update biases based on the gradient.

    def calculate_gradient(self, predictions, labels):
        """
        Find the gradient matrix given the predictions and labels.
        """
        pass
