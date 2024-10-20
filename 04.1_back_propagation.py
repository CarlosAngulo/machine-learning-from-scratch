import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

W = np.random.normal(scale=0.5, size=(2, 1))


def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

z = np.dot(inputs, W)

# output = sigmoid(z)
# print(output)

# Not eveb close to what we want which is
# [[0.5       ]
#  [0.34212528]
#  [0.49682545]
#  [0.33927294]]

# function to measure the output error: error cuadrático medio
def mse(output, target):
    return np.mean((output - target) ** 2) / 2


# error = 1/2 (output - y) ** 2
# error = 1/2 (sigmoid(z) - y) ** 2
# error = 1/2 (sigmoid(np.dot(inputs, W)) - y) ** 2
# We are going to derivate respect to W. Appilying the Chain rule:
# error' = (output - y) * sigmoid(z) * (1 - sigmoid(z)) * inputs

output = sigmoid(z)
error = output - y
delta = error * (sigmoid(z) * (1 - sigmoid(z)))
# We need to transpose inputs
derivada = np.dot(inputs.T, delta)
print(derivada)

