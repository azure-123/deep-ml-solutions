import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	return scores - np.log(np.sum(np.exp(scores)))

A = np.array([1, 2, 3])
print(log_softmax(A))