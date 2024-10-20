import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

weights = np.random.normal(scale=0.5, size=(2, 1))

# Sigmoid function that will be used to train the neuron
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

# The following code creates a list of 50 points between -10 and 10 to plot the sigmoid function.
x = np.linspace(-10, 10, 50)

plt.plot(x, sigmoid(x))
plt.grid(True)

z = np.dot(inputs, weights)

output = sigmoid(z)
print(output)

plt.plot(inputs, output, 'r.')
plt.show()

# The output is not what we need yet:
# [[0.5       ]
#  [0.63815746]
#  [0.57200855]
#  [0.70212273]]

# So the next step is preparing the backpropagation process