from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.show()


def show_error_samples(X, y_true, y_pred, class_names, num_samples=5):
    errors = np.where(y_true != y_pred)[0]
    sample_indices = np.random.choice(errors, num_samples)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[:,idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}')
        plt.axis('off')
    plt.show()


def plot_feature_importance(weights, feature_names, top_n=20):
    importance = np.mean(np.abs(weights), axis=1)
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance (First Layer Weights)')
    plt.barh(range(top_n), importance[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Mean Absolute Weight')
    plt.show()


def plot_class_distribution(y_true, y_pred, classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # True distribution
    sns.countplot(x=y_true, ax=ax1, order=classes)
    ax1.set_title('True Class Distribution')

    # Predicted distribution
    sns.countplot(x=y_pred, ax=ax2, order=classes)
    ax2.set_title('Predicted Class Distribution')

    plt.show()


def visualize_first_layer_weights(weights, input_shape=(28, 28), n_cols=8):
    num_neurons = weights.shape[0]
    weight_images = weights.T.reshape(num_neurons, *input_shape)

    n_rows = int(np.ceil(num_neurons / n_cols))

    plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    vmax = np.max(np.abs(weight_images))
    vmin = -vmax

    for i in range(num_neurons):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(weight_images[i], cmap='RdBu', vmin=vmin, vmax=vmax)
        plt.title(f'Neuron {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def view_model_prediction_curve(model, sample_frequency=0.01, start=0, stop=10):
    num_samples = int((stop - start) / sample_frequency)
    X = np.linspace(start, stop, num_samples).reshape(1, -1)
    Y = model.forward(X).reshape(-1, 1)
    X = X.reshape(-1, 1)
    print(Y)
    plt.figure(figsize=(10, 5))
    plt.plot(X, Y, label='Model Prediction Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, num_classes, sample_frequency=0.01, start=(0, 0), stop=(10, 10)):
    num_samples_x = int((stop[0] - start[0]) / sample_frequency)
    num_samples_y = int((stop[1] - start[1]) / sample_frequency)
    samples_x = np.linspace(start[0], stop[0], num_samples_x).reshape(1, -1)
    samples_y = np.linspace(start[1], stop[1], num_samples_y).reshape(1, -1)

    predictions = []
    plt.figure(figsize=(10, 5))

    for i in range(0, num_samples_x):
        for j in range(0, num_samples_y):
            prediction = np.squeeze(model.predict(np.array([samples_x[:, i], samples_y[:, j]])))
            predictions.append(prediction)
    predictions = np.array(predictions).reshape(int((stop[0] - start[0])/sample_frequency), int((stop[1] - start[1])/sample_frequency))
    im = plt.imshow(predictions, cmap=plt.cm.get_cmap('viridis', num_classes))

    cbar = plt.colorbar(im, ticks=np.arange(num_classes))
    cbar.set_label('Class')

    plt.title('Decision Boundary of the Model')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Note that this function doesn't show an accurate indication of how much the input pixels affect the output, but only
# provides a linear approximation (activation functions are non-linear, their influence can't be visualized directly)
def visualize_effective_weights_for_layer(weights_list, target_layer_index, input_shape=(28, 28), n_cols=8):
    if not weights_list or target_layer_index < 0 or target_layer_index >= len(weights_list):
        print(f"Error: Invalid target_layer_idx ({target_layer_index}) or empty weights_list.")
        return
    if not isinstance(weights_list, list) or not all(isinstance(w, np.ndarray) for w in weights_list):
         print("Error: weights_list must be a list of NumPy arrays.")
         return

    print(f"Calculating effective weights for Layer {target_layer_index + 1} (index {target_layer_index})...")

    effective_weights = weights_list[0]
    print(f"  Layer 0 (W1) shape: {effective_weights.shape}")


    for i in range(1, target_layer_index + 1):
        W_next = weights_list[i]
        print(f"  Layer {i} (W{i+1}) shape: {W_next.shape}")

        if effective_weights.shape[0] != W_next.shape[1]:
             print(f"Error: Shape mismatch for matrix multiplication.")
             print(f"  Cannot multiply effective_weights shape {effective_weights.shape} with W{i+1} shape {W_next.shape}")
             print(f"  Expected W{i+1} to have shape ({W_next.shape[0]}, {effective_weights.shape[0]})")

        effective_weights = np.dot(W_next, effective_weights)
        print(f"  Effective weights shape after Layer {i}: {effective_weights.shape}")

    num_neurons = effective_weights.shape[0]
    num_input_features = effective_weights.shape[1]
    expected_input_features = np.prod(input_shape)

    if num_input_features != expected_input_features:
        print(f"Error: Number of input features in effective weights ({num_input_features})")
        print(f"       does not match expected input features from input_shape ({expected_input_features}).")

    print(f"\nVisualizing effective weights for {num_neurons} neurons in Layer {target_layer_index + 1}.")

    weight_images = effective_weights.reshape(num_neurons, *input_shape)

    n_rows = int(np.ceil(num_neurons / n_cols))

    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    plt.suptitle(f'Effective Input Weights for Neurons in Layer {target_layer_index + 1} (Linear Approx.)', fontsize=16)

    vmax = np.max(np.abs(weight_images))
    vmin = -vmax

    for i in range(num_neurons):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(weight_images[i], cmap='RdBu', vmin=vmin, vmax=vmax)
        plt.title(f'Neuron {i}')
        plt.axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()
