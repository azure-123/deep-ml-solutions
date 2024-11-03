import math

def softmax(scores: list[float]) -> list[float]:
	# Your code here
	prob_sum = sum([math.exp(score) for score in scores])
	probabilities = [round(math.exp(score) / prob_sum, 4) for score in scores]
	return probabilities

scores = [1, 2, 3]
print(softmax(scores))