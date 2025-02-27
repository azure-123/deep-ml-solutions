import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))
		
	def sigmoid(self, input):
		return 1 / (1 + np.exp(-input))
	
	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		h_t = initial_hidden_state
		c_t = initial_cell_state
		outputs = []
		for x_t in x:
			x_t = x_t.reshape(-1, 1)
			concat = np.vstack((h_t, x_t))

			f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
			i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
			c_tilde_t = np.tanh(np.dot(self.Wc, concat) + self.bc)
			c_t = f_t * c_t + i_t * c_tilde_t
			o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
			h_t = o_t * np.tanh(c_t)
			outputs.append(o_t)
		return outputs, h_t, c_t

input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(input_size=1, hidden_size=1)
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

print(final_h)