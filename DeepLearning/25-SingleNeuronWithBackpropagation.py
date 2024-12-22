import numpy as np
def sigmoid(input: np.ndarray):
    output = np.array([1 / (1 + np.exp(-element)) for element in input])
    return output

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int):
    features = np.array(features)
    labels = np.array(labels)
    initial_weights = np.array(initial_weights)
    mse_values = []
    for _ in range(epochs):
        # prediction
        raw_output = np.dot(initial_weights, features.T) + initial_bias
        predictions = sigmoid(raw_output)
        mse = round(np.sum((1 / len(labels)) * (predictions - labels)**2), 4)
        mse_values.append(mse)

        # back propagation
        weight_grad = (2 / len(labels)) * np.dot(features.T, (predictions - labels) * predictions * (1 - predictions))
        bias_grad = (2 / len(labels)) * np.sum(predictions * (1 - predictions) * (predictions - labels))

        # update
        initial_weights -= learning_rate * weight_grad
        initial_bias -= learning_rate * bias_grad
        updated_weights = np.round(initial_weights, 4)
        updated_bias = np.round(initial_bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values

features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
labels = [1, 0, 0]
initial_weights = [0.1, -0.2]
initial_bias = 0.0
learning_rate = 0.1
epochs = 2
print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))