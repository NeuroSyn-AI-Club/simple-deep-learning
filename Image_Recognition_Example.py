# Main Libraries
import numpy as np
from sklearn.datasets import fetch_openml
from Activations import *
from Model import Model
from Initializers import *
from Objectives import *
from Optimizers import *
from Metrics_and_Visualizations import *

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Data Pre-processing and labels
X, y = mnist["data"], mnist["target"]
m = y.shape[0]
y_vector = np.zeros((10, m))
for i in range(m):
    y_vector[int(y[i]), i] = 1

X_reshaped = X.T.reshape(784, -1)
X_reshaped = X_reshaped / 255.0 # Normalize the image values (very important)

X_train = X_reshaped[:, :60000]  # (784, 55000)
X_test = X_reshaped[:, 60000:]   # (784, 10000)
X_dev = X_reshaped[:, 55000:60000] # (784, 5000)

y_train = y_vector[:, :60000]  # (10, 55000)
y_test = y_vector[:, 60000:].astype(np.int64)  # (10, 10000)
y_dev = y_vector[:, 55000:60000] # (784, 5000)
y_test_labels = np.astype(y[60000:], np.int64)

# Set Model Hyperparameters
np.random.seed(244)
network_shape = [784, 192, 96, 48, 10]
activations = [LeakyReLU(), LeakyReLU(), LeakyReLU(), Sigmoid()]
initializer = HeInitializer()
objective = MeanSquaredError()
optimizer = AdamOptimizer(0.00074, 0.9, 0.999, 1e-8)


neural_network = Model(network_shape, activations, initializer, objective, optimizer, lambd=0.001)
costs, accuracies, dev_costs, dev_accuracies = neural_network.train(X_train, y_train, 2, 32, print_cost_every=1, X_dev=X_dev, Y_dev=y_dev)
predictions, accuracy, correct_matches, false_matches = neural_network.predict(X_test, y_test)

loss:dict[str, int] = dict(loss=costs, val_loss=dev_costs, accuracy=accuracies, val_accuracy=dev_accuracies)
weights = [ layer.weights for layer in neural_network.layers ]

# See Metrics
print(f"Accuracy: {100*accuracy}%")
print(f"Number of Correct Predictions: {correct_matches}")
print(f"Number of False Predictions: {false_matches}")
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Visualizations
plot_training_history(loss)
plot_confusion_matrix(y_test_labels, predictions, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
show_error_samples(X_test, y_test_labels, predictions, class_names, 15)
plot_class_distribution(y_test_labels, predictions, class_names)
plot_feature_importance(neural_network.layers[0].weights, [str(i//28) + ", " + str(i%28) for i in range(784)], 50)
visualize_first_layer_weights(neural_network.layers[0].weights)
visualize_effective_weights_for_layer(weights, target_layer_index=len(neural_network.layers)-1, input_shape=(28, 28), n_cols=8)