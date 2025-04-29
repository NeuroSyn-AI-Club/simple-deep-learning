import numpy as np

class Objective:
    def cost(self, AL, Y):
        pass

    def cost_prime(self, AL, Y):
        pass


class MeanSquaredError:
    def cost(self, AL, Y):
        cost = np.squeeze(np.sum(np.mean(np.square(AL - Y), axis=1, keepdims=False))) / AL.shape[0]
        cost = np.squeeze(cost)
        return cost

    def cost_prime(self, AL, Y):
        m = Y.shape[1]
        cost_derivative = -2 / m * (Y - AL)
        return cost_derivative


class CrossEntropyLoss:
    def __init__(self, use_softmax=True):
        self.use_softmax = use_softmax

    def softmax(self, A):
        e_A = np.exp(A)
        return e_A / np.sum(e_A, axis=0, keepdims=False)

    def cost(self, AL, Y):
        A = self.softmax(AL) if self.use_softmax else AL
        m = Y.shape[1]
        cost = -np.sum(Y*np.log(A)) / m
        cost = np.squeeze(cost)
        return cost

    def cost_prime(self, AL, Y):
        m = Y.shape[1]
        if self.use_softmax:
            return AL - Y
        else:
            return -Y/(AL * m)


def compute_frobenius_norm(W, lambd, m):
    if lambd == 0:
        return 0
    frobenius_norm = np.sum([ np.sum(np.square(w)) for w in W])
    return frobenius_norm * lambd / m