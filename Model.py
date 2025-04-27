import copy
import numpy as np

from Utils import random_mini_batches

class Layer:
    def __init__(self, num_neurons, num_neurons_previous, activation_function, initialization, optimizer_template):
        self.weights = initialization.initialize_weights(num_neurons, num_neurons_previous)
        self.biases = initialization.initialize_biases(num_neurons) # Assuming consistent naming

        weights_shape = self.weights.shape
        biases_shape = self.biases.shape

        optimizer_instance = copy.deepcopy(optimizer_template)
        optimizer_instance.build(weights_shape, biases_shape)
        self.optimizer = optimizer_instance

        self.activation_function = activation_function
        self.activation_previous = None
        self.weighted_sum = None
        self.weight_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.biases)


    def forward_propagate(self, A_prev):
        self.activation_previous = A_prev
        self.weighted_sum = np.dot(self.weights, A_prev) + self.biases
        A = self.activation_function.activation(self.weighted_sum)
        return A


    def backward_propagate(self, dA):
        if self.activation_previous is None:
             raise RuntimeError("Forward pass must be run before backward pass.")

        m = self.activation_previous.shape[1]
        if m == 0:
              self.weight_gradient = np.zeros_like(self.weights)
              self.bias_gradient = np.zeros_like(self.biases)
              dZ_zero = np.zeros((self.weights.shape[0], 0))
              dA_prev = np.dot(self.weights.T, dZ_zero)
              return dA_prev

        dZ = dA * self.activation_function.derivation(self.weighted_sum)
        self.weight_gradient = np.dot(dZ, self.activation_previous.T)
        self.bias_gradient = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)
        self.update_parameters()
        return dA_prev


    def update_parameters(self):
        m = self.weighted_sum.shape[1]
        if self.optimizer is None:
            return
        update_weights, update_biases = self.optimizer.update_parameters(self.weight_gradient, self.bias_gradient)
        self.weights += update_weights
        self.biases += update_biases


    def get_parameters(self):
        return self.weights, self.biases


class Model:
    def __init__(self, network_shape, activations, initialization, objective, optimizer):
        if len(activations) != len(network_shape) - 1:
            raise RuntimeError("Length of activations list must be len(network_shape) - 1")

        self.shape = network_shape
        self.activations = activations
        self.initialization = initialization
        self.objective = objective
        self.optimizer_template = optimizer

        self.layers = self.initialize_layers()

    def initialize_layers(self):
        layers = []
        num_weight_layers = len(self.shape) - 1
        for i in range(num_weight_layers):
            num_neurons_previous = self.shape[i]
            num_neurons = self.shape[i+1]
            activation_function = self.activations[i]
            optimizer_instance = copy.deepcopy(self.optimizer_template)
            optimizer_instance.build([self.shape[i+1], self.shape[i]], [self.shape[i+1], 1])

            layer = Layer(num_neurons, num_neurons_previous,
                          activation_function,
                          self.initialization,
                          optimizer_instance)
            layers.append(layer)
        return layers

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward_propagate(A)
        return A

    def backward(self, dAL):
        dA_prev = dAL
        for layer in reversed(self.layers):
            dA_prev = layer.backward_propagate(dA_prev)

    def update_model_parameters(self):
        for layer in self.layers:
            layer.update_parameters()

    def train(self, X, Y, epochs, batch_size, print_cost_every=0, X_dev=None, Y_dev=None):
        if X.shape[0] != self.shape[0]:
            raise ValueError("The features of the training data must equal the number of neurons in the first layer")
        costs = []
        accuracies = []
        dev_costs = []
        dev_accuracies = []
        m = X.shape[1]

        # Iterate over epochs
        for i in range(epochs):
            mini_batches = random_mini_batches(X, Y, batch_size)
            epoch_cost_total = 0
            epoch_accuracy_total = 0

            if X_dev is not None:
                dev_AL = self.forward(X_dev)
                dev_predictions = np.argmax(dev_AL, axis=0)
                dev_true_labels = np.argmax(Y_dev, axis=0)
                dev_epoch_cost = self.objective.cost(dev_AL, Y_dev) 
                dev_epoch_accuracy = np.sum(dev_predictions == dev_true_labels) / X_dev.shape[1]
                dev_costs.append(dev_epoch_cost)
                dev_accuracies.append(dev_epoch_accuracy)

            # Iterate over all mini_batches
            for X_batch, Y_batch in mini_batches:
                if X_batch.shape[1] == 0: continue

                # Forward Propagate
                AL = self.forward(X_batch)

                # Calculate Train, Validation(Dev set) costs and accuracies
                predictions = np.argmax(AL, axis=0)
                true_labels = np.argmax(Y_batch, axis=0)
                batch_cost = self.objective.cost(AL, Y_batch)
                batch_accuracy = np.sum(predictions == true_labels)
                epoch_cost_total += batch_cost * X_batch.shape[1]
                epoch_accuracy_total += batch_accuracy

                # Calculate objective derivative and backpropagate (parameters are updated in layers individually)
                dAL = self.objective.cost_prime(AL, Y_batch)

                self.backward(dAL)

            # Keep cost and accuracy history
            cost_avg = epoch_cost_total / m
            accuracy_avg = epoch_accuracy_total / m
            costs.append(cost_avg)
            accuracies.append(accuracy_avg)

            # If a dev set is passed, calculate accuracy and costs


            if print_cost_every > 0 and (i % print_cost_every == 0 or i == epochs - 1):
                print(f"Epoch {i}:")
                print(f"  Training Cost: {cost_avg}")
                print(f"  Training Accuracy: {accuracy_avg*100}%")
                if X_dev is not None:
                    print(f"  Validation Cost: {dev_epoch_cost}")
                    print(f"  Validation Accuracy: {dev_epoch_accuracy*100}%")

        if X_dev is not None:
            return costs, accuracies, dev_costs, dev_accuracies
        else:
            return costs, accuracies

    def predict(self, X, Y=None):
        AL = self.forward(X)
        predictions = np.argmax(AL, axis=0)

        accuracy = None
        correct_guesses = None
        false_guesses = None
        if Y is not None:
            true_labels = np.argmax(Y, axis=0)
            correct_guesses = np.sum(predictions == true_labels)
            total_samples = Y.shape[1]
            if total_samples > 0:
                 accuracy = correct_guesses / total_samples
                 false_guesses = total_samples - correct_guesses
            else:
                 accuracy = 0
                 false_guesses = 0

        if Y is not None:
            return predictions, accuracy, correct_guesses, false_guesses
        else:
            return predictions
