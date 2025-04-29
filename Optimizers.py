import numpy as np


class GradientDescent:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate


    def build(self, _, __):
        return


    def update_parameters(self, weights_gradient, bias_gradient):
        update_W = -self.learning_rate * weights_gradient
        update_b = -self.learning_rate * bias_gradient
        return update_W, update_b


class MomentumOptimizer:
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self._built = False

        self.moment_W = 0
        self.moment_b = 0


    def build(self, weights_shape, biases_shape):
        if self._built:
            return  # Avoid re-initialization
        self.moment_W = np.zeros(weights_shape)
        self.moment_b = np.zeros(biases_shape)
        self._built = True


    def update_parameters(self, weights_gradient, bias_gradient):
        if not self._built:
            raise RuntimeError("Optimizer state not initialized. Call build() before update_parameters().")
        self.moment_W = self.beta * self.moment_W + (1 - self.beta) * weights_gradient
        self.moment_b = self.beta * self.moment_b + (1 - self.beta) * bias_gradient
        update_W = -self.learning_rate * self.moment_W
        update_b= -self.learning_rate * self.moment_b
        return update_W, update_b


class AdamOptimizer:
    def __init__(self, learning_rate, beta1, beta2, epsilon):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self._built = False

        self.moment1_W = 0
        self.moment1_b = 0
        self.moment2_W = 0
        self.moment2_b = 0

    def build(self, weights_shape, biases_shape):
        if self._built:
            return  # Avoid re-initialization

        self.moment1_W = np.zeros(weights_shape)
        self.moment1_b = np.zeros(biases_shape)
        self.moment2_W = np.zeros(weights_shape)
        self.moment2_b = np.zeros(biases_shape)
        self._built = True

    def update_parameters(self, weights_gradient, bias_gradient):
        if not self._built:
            raise RuntimeError("Optimizer state not initialized. Call build() before update_parameters().")
        self.timestep += 1

        self.moment1_W = self.beta1 * self.moment1_W + (1 - self.beta1) * weights_gradient
        self.moment1_b = self.beta1 * self.moment1_b + (1 - self.beta1) * bias_gradient

        self.moment2_W = self.beta2 * self.moment2_W + (1 - self.beta2) * np.square(weights_gradient)
        self.moment2_b = self.beta2 * self.moment2_b + (1 - self.beta2) * np.square(bias_gradient)

        corrected_moment1_W = self.moment1_W / (1 - np.power(self.beta1, self.timestep))
        corrected_moment1_b = self.moment1_b / (1 - np.power(self.beta1, self.timestep))

        corrected_moment2_W = self.moment2_W / (1 - np.power(self.beta2, self.timestep))
        corrected_moment2_b = self.moment2_b / (1 - np.power(self.beta2, self.timestep))
        update_W = -self.learning_rate * corrected_moment1_W / (np.sqrt(corrected_moment2_W) + self.epsilon)
        update_b = -self.learning_rate * corrected_moment1_b / (np.sqrt(corrected_moment2_b) + self.epsilon)

        return update_W, update_b


class RMSprop:
    def __init__(self, learning_rate, beta, epsilon):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.timestep = 0
        self._built = False

        self.squared_W = 0
        self.squared_b = 0

    def build(self, weights_shape, biases_shape):
        if self._built:
            return  # Avoid re-initialization

        self.squared_W = np.zeros(weights_shape)
        self.squared_b = np.zeros(biases_shape)
        self._built = True

    def update_parameters(self, weights_gradient, bias_gradient):
        if not self._built:
            raise RuntimeError("Optimizer state not initialized. Call build() before update_parameters().")
        self.timestep += 1

        self.squared_W = self.beta * self.squared_W + (1 - self.beta) * np.square(weights_gradient)
        self.squared_b = self.beta * self.squared_b + (1 - self.beta) * np.square(bias_gradient)

        # Unsure if bias correction is a part of the original RMSprop algorithm, but I don't think it would hurt.
        corrected_squared_W = self.squared_W / (1 - np.power(self.beta, self.timestep))
        corrected_squared_b = self.squared_b / (1 - np.power(self.beta, self.timestep))
        update_W = -self.learning_rate * weights_gradient / (np.sqrt(corrected_squared_W) + self.epsilon)
        update_b = -self.learning_rate * bias_gradient / (np.sqrt(corrected_squared_b) + self.epsilon)

        return update_W, update_b