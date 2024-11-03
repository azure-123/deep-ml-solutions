import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    h = np.array(initial_hidden_state)
    wx_array = np.array(Wx)
    wh_array = np.array(Wh)
    b_array = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(wx_array, x) + np.dot(wh_array, h) + b_array)
    final_hidden_state = np.round(h, 4)
    return final_hidden_state.tolist()

input_sequence = [[1.0], [2.0], [3.0]]
initial_hidden_state = [0.0]
Wx = [[0.5]]  # Input to hidden weights
Wh = [[0.8]]  # Hidden to hidden weights
b = [0.0]     # Bias
print(rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b))