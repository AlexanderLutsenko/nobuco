import numpy as np

x_torch = np.asarray([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
x_keras = x_torch.transpose()
print('x_keras:\n', x_keras)