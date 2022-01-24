import numpy as np
from utils import sigmoid, initialize_weights, initialize_biases, softmax


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.layer_size = layer_sizes
        self.weights: np.array = initialize_weights(layer_sizes)
        self.biases: np.array = initialize_biases(layer_sizes)

    def normalization(self):
        for i, layer_weight in enumerate(self.weights):
            layer_weight_mean = np.mean(layer_weight)
            layer_weight_var = np.var(layer_weight)

            self.weights[i] = (layer_weight - layer_weight_var) / layer_weight_mean

        for layer_bias in self.biases:
            layer_bias_mean = np.mean(layer_bias)
            layer_bias_var = np.var(layer_bias)

            self.biases[i] = (layer_bias - layer_bias_var) / layer_bias_mean

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return sigmoid(x)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        a = x
        for w, b in list(zip(self.weights, self.biases))[:-1]:
            z = np.dot(w, a) + b
            a = self.activation(z)

        # Using Softmax on the last layer
        w, b = self.weights[-1], self.biases[-1]
        z = np.dot(w, a) + b
        a = softmax(z)

        return a
