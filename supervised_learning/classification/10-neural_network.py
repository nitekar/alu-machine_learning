#!/usr/bin/env python3
"""Neural Network for Binary Classification"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network
    with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Constructor for NeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = self.sigmoid(Z1)

        Z2 = np.dot(self.W2, self.__A1) + self.b2
        self.__A2 = self.sigmoid(Z2)

        return self.__A1, self.__A2
    