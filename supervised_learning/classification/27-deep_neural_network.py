#!/usr/bin/env python3
"""Deep Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network Class"""

    def __init__(self, nx, layers):
        """Constructor for DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize layers
        layer_sizes = [nx] + layers
        for l in range(1, self.L + 1):
            layer_size = layer_sizes[l]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            weight_key = 'W' + str(l)
            bias_key = 'b' + str(l)
            self.weights[weight_key] = np.random.randn(
                layer_size, layer_sizes[l - 1]) * np.sqrt(
                    2 / layer_sizes[l - 1])
            self.weights[bias_key] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Forward prop"""
        self.__cache['A0'] = X
        A = X

        for layer in range(1, self.__L + 1):
            Z = np.dot(self.__weights[
                'W' + str(layer)], A) + self.__weights['b' + str(layer)]
            if layer == self.__L:
                # Use softmax activation for the output layer
                exp_z = np.exp(Z)
                A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
            else:
                A = self.sigmoid(Z)
            self.__cache['A' + str(layer)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        epsilon = 1e-15
        # Calculate categorical cross-entropy cost
        cost = -np.sum(Y * np.log(A + epsilon)) / m
        return cost

    def evaluate(self, X, Y):
        """EValuate DNN"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        one_hot_predictions = np.eye(Y.shape[0])[predictions].T
        cost = self.cost(Y, A)
        return one_hot_predictions, cost

    def sigmoid_derivative(self, A):
        """Derivative of the sigmoid activation function"""
        return A * (1 - A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent on the neural network"""
        m = Y.shape[1]
        A = cache["A" + str(self.L)]
        dZ = A - Y

        for l in reversed(range(1, self.L + 1)):
            A_prev = cache["A" + str(l - 1)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                W = self.weights["W" + str(l)]
                dZ = np.dot(W.T, dZ) * self.sigmoid_derivative(A_prev)

            self.weights["W" + str(l)] -= alpha * dW
            self.weights["b" + str(l)] -= alpha * db

    def train(self,
              X,
              Y,
              iterations=5000,
              alpha=0.05,
              verbose=True,
              graph=True,
              step=100):
        """DNN Training"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not verbose and not graph or step > iterations:
            step = iterations
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
        