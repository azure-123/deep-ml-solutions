import numpy as np
def recall(y_true, y_pred):
    true_positive = np.sum(y_true & y_pred == 1)
    false_negative = np.sum(y_true & ~y_pred == 1)
    return 0.0 if true_positive + false_negative == 0 else round(true_positive / (true_positive + false_negative), 3)

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))