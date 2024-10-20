import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

W = np.random.normal(scale=0.5, size=(2, 1))

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

z = np.dot(inputs, W)

output = sigmoid(z)
print(W)
#print(y)
print(output)

# Not even close to what we want which is [[0, 0], [0, 1], [1, 0], [1, 1]]
