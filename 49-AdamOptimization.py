import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	m_t = np.zeros_like(x0)
	v_t = np.zeros_like(x0)
	for t in range(1, num_iterations + 1):
		grad_x = grad(x0)
		m_t = beta1 * m_t + (1 - beta1) * grad_x
		v_t = beta2 * v_t + (1 - beta2) * (grad_x ** 2)
		m_hat = m_t / (1 - np.power(beta1, t))
		v_hat = v_t / (1 - np.power(beta2, t))
		x0 -= learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon))
	return x0
def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)

