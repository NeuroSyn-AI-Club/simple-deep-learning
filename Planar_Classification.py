import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Activations import *
from Initializers import *
from Objectives import *
from Optimizers import *
from Model import Model
from Metrics_and_Visualizations import *

N = 200 # number of points per class
D = 2 # dimensionality
K = 6 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=matplotlib.colormaps["viridis"])
plt.show()

X_train = X.T
y_train = y.reshape(1, y.shape[0])
y_train_onehot = np.zeros((y_train.size, y_train.max() + 1))
y_train_onehot[np.arange(y_train.size), y_train] = 1
y_train_onehot = y_train_onehot.T


# Network architecture
network_shape = [2, 8, 8, 8, 8, 8, 8, K]
activations = [ELU(), ELU(), ELU(), ELU(), ELU(), ELU(),  Identity()]
initializer = HeInitializer()
objective = CrossEntropyLoss()
optimizer = AdamOptimizer(learning_rate=0.0028,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-8)

# Create and train model
neural_network = Model(network_shape, activations, initializer, objective, optimizer)
costs, accuracies = neural_network.train(X_train, y_train_onehot,
                                       epochs=2000,
                                       batch_size=32,
                                       print_cost_every=100)

plot_decision_boundary(neural_network, num_classes=K, sample_frequency=0.005, start=(-1, -1), stop=(1, 1))