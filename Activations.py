import numpy as np

class Sigmoid:
    def activation(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def derivation(self, Z):
        return self.activation(Z) * (1 - self.activation(Z))

class ReLU:
    def activation(self, Z):
        A = np.maximum(Z, 0)
        return A

    def derivation(self, Z):
        Z_prime = np.where(Z > 0, 1, 0)
        return Z_prime

class LeakyReLU:
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def activation(self, Z):
        A = np.maximum(Z, Z*self.negative_slope)
        return A

    def derivation(self, Z):
        Z_prime = np.where(Z > 0, 1, self.negative_slope)
        return Z_prime

class Tanh:
    def activation(self, Z):
        return np.tanh(Z)

    def derivation(self, Z):
        Z_prime = 1 - np.tanh(Z) ** 2
        return Z_prime

class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def activation(self, Z):
        A = np.where(Z > 0, Z, self.alpha * (np.exp(Z) - 1))
        return A

    def derivation(self, Z):
        Z_prime = np.where(Z > 0, 1, self.alpha * np.exp(Z))
        return Z_prime

class Swish:
    def __init__(self, beta=1.0):
        self.beta = beta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.beta * x))

    def activation(self, Z):
        A = Z * self.sigmoid(Z)
        return A

    def derivation(self, Z):
        sigmoid_Z = self.sigmoid(Z)
        Z_prime = sigmoid_Z + Z * sigmoid_Z * (1 - sigmoid_Z) * self.beta
        return Z_prime


class Identity:
    def activation(self, Z):
        return Z

    def derivation(self, Z):
        return 1