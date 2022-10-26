import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt


def one_hot_encoder(arr: np.ndarray):
    enc = OneHotEncoder()
    return enc.fit_transform(arr).toarray()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return 1. * (z > 0)

class neuralnetwork:

    def __init__(self, layers, activation, activation_derivative,
                 learning_rate=0.1, epsilon=1e-8, batch_size=100, max_epochs=1000, adaptive=False, verbose=False):
        self.layers = layers
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.adaptive = adaptive
        self.verbose = verbose
        self.weights = []
        self.biases = []
        self.losses = []
        self.accuracies = []
        self.confusion_matrices = []
        # self.init_weights_biases()
    def init_weights_biases(self):
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(
                self.layers[i], self.layers[i - 1]) * (2 / self.layers[i - 1]) ** 0.5)
            self.biases.append(np.random.randn(
                self.layers[i], 1) * (2 / self.layers[i - 1]) ** 0.5)
                
    def forward_propagation(self, X):
        self.output = [X]
        for i in range(len(self.weights)-1):
            net = np.dot(self.weights[i], self.output[i]) + self.biases[i]
            self.output.append(self.activation(net))
        net = np.dot(self.weights[-1], self.output[-1]) + self.biases[-1]
        self.output.append(sigmoid(net))
        return self.output[-1]

    def back_propagation(self, X, y):
        m = X.shape[1]
        self.forward_propagation(X)
        delta = [None] * len(self.weights)
        delta[-1] = (self.output[-1] - y) * sigmoid_derivative(self.output[-1])
        for i in range(len(self.weights)-2, -1, -1):
            delta[i] = np.dot(self.weights[i+1].T, delta[i+1]) * self.activation_derivative(self.output[i+1])
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * \
                np.dot(delta[i], self.output[i].T) / m
            self.biases[i] -= self.learning_rate * np.sum(
                delta[i], axis=1, keepdims=True) / m

    def predict(self, X):
        return self.forward_propagation(X.T).T
    
    def score(self, X, y):
        return np.mean(np.argmax(self.predict(X), axis=0) == y)
    
    def fit(self, X, y):
        y = one_hot_encoder(y.reshape(-1, 1))
        self.layers = [X.shape[1]] + self.layers + [y.shape[1]]
        self.init_weights_biases()
        m = X.shape[0]
        for epoch in range(self.max_epochs):
            for i in range(0, m, self.batch_size):
                self.back_propagation(X[i:i+self.batch_size,:].T, y[i:i+self.batch_size,:].T)
            if self.adaptive:
                self.learning_rate = self.learning_rate / (1+epoch)**0.5
            if self.verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}")
                    print(f"Loss: {self.loss(X, y)}")
                    print(f"Accuracy: {self.accuracy(X, y)}")
            