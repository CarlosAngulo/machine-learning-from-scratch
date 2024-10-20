import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

weights = np.random.normal(scale=0.5, size=(2, 1))

output = np.dot(inputs, weights)

print(output)

#For now, the weights are random numbers like the following:
#[[-0.07725875]
# [ 0.45725448]]

# And the output will be something like this:
# [[ 0.        ]
#  [-0.68402811]
#  [ 0.96161534]
#  [ 0.27758723]]

# The results should be close to the vector y: [[0], [0], [0], [1]]
# So we are going to train the neuron in the next files.