import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float): # type: ignore
	# Your code here
	neuron_output = []
	for feature in features:
		neuron_output.append(sum([f * w for f, w in zip(feature, weights)]) + bias)
	probabilities = [round(1 / (1 + math.exp(-neuro_out)), 4) for neuro_out in neuron_output]
	mse = sum([(p - l) ** 2 for p, l in zip(probabilities, labels)]) / len(labels)
	return probabilities , round(mse, 4)

features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1

print(single_neuron_model(features, labels, weights, bias))