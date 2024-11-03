import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    matrix = pad(input_matrix, padding)
    input_height, input_width = matrix.shape
    kernel_height, kernel_width = kernel.shape
    output_matrix = []
    for i in range(kernel_height - 1, input_height, stride):
        row_vector = []
        for j in range(kernel_width - 1, input_width, stride):
            row_vector.append(np.sum(kernel * matrix[i - kernel_height + 1: i + 1, j - kernel_width + 1: j + 1]))
        output_matrix.append(row_vector)
    return np.array(output_matrix)

def pad(input_matrix, padding):
	matrix = input_matrix.tolist()
	pad_list = [0 for i in range(len(matrix[0]) + padding * 2)]
	for i in range(len(matrix)):
		matrix[i] = [0 for _ in range(padding)] + matrix[i] + [0 for _ in range(padding)]
	matrix = [pad_list for _ in range(padding)] + matrix + [pad_list for _ in range(padding)]
	return np.array(matrix)
    
input_matrix = np.array([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.],
])
kernel = np.array([
    [.5, 3.2],
    [1., -1.],
])
padding, stride = 2, 2

print(simple_conv2d(input_matrix, kernel, padding, stride))