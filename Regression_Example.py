import numpy as np
import matplotlib.pyplot as plt
from Activations import *
from Model import Model
from Initializers import *
from Objectives import *
from Optimizers import *
from Metrics_and_Visualizations import *


def generate_sine_function(num_points=1000, amplitude=1.0, frequency=0.5,
                           phase=0.0, noise_level=0.1, x_range=(0, 2 * np.pi)):
    x = np.linspace(x_range[0], x_range[1], num_points).reshape(1, -1)
    y_clean = amplitude * np.sin(2 * np.pi * frequency * x + phase)

    noise = np.random.normal(scale=noise_level, size=y_clean.shape)
    y = y_clean + noise

    return x, y, y_clean


# Generate data with noise
x, y, y_clean = generate_sine_function(noise_level=0.05, amplitude=0.4, num_points=1000)

# Plot the data
plt.figure(figsize=(10, 5))
plt.scatter(x.ravel(), y.ravel(), s=5, label='Noisy data')
plt.plot(x.ravel(), y_clean.ravel(), 'r-', label='True sine wave')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Sine Function')
plt.legend()
plt.grid(True)
plt.show()

# Network architecture
network_shape = [1, 5, 5, 1]
activations = [Swish(), Swish(), Identity()]
initializer = HeInitializer()
objective = MeanSquaredError()
optimizer = AdamOptimizer(learning_rate=0.0074, beta1=0.9, beta2=0.999, epsilon=1e-8)  # Increased learning rate

# Create and train model
neural_network = Model(network_shape, activations, initializer, objective, optimizer)
costs, accuracies = neural_network.train(x, y, epochs=1500, batch_size=10, print_cost_every=100)


# View predictions
def view_model_prediction_curve(model, sample_frequency=0.01, start=0, stop=2 * np.pi):
    """Plot model predictions over a range of x values"""
    num_samples = int((stop - start) / sample_frequency)
    X = np.linspace(start, stop, num_samples).reshape(1, -1)
    Y_pred = model.forward(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(x.ravel(), y.ravel(), s=5, label='Training data')
    plt.plot(X.ravel(), Y_pred.ravel(), 'g-', linewidth=2, label='Model predictions')
    plt.plot(x.ravel(), y_clean.ravel(), 'r--', label='True sine wave')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Prediction vs True Function')
    plt.legend()
    plt.grid(True)
    plt.show()


view_model_prediction_curve(neural_network)