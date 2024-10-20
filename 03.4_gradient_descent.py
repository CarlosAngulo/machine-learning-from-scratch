import matplotlib.pyplot as plt
import numpy as np

def my_function(x):
    return x**2 - 5*x + 2

def derivada(x):
    return 2*x - 5

x = np.linspace(-20, 20, 100)
y = my_function(x)

plt.plot(x, y)

############
# From this point on, we are using the approach of grandient descent

# define the leanring rate which multiplies the dx
lr = 0.1
init_x = -15 #initial point x
init_y = my_function(init_x) #initial point y
dx = derivada(init_x)
px = init_x - dx  * lr
py = my_function(px)
plt.plot(init_x, init_y, 'r.')
plt.plot(px, py, 'r.')

# iterates several times and the px is decreasing according to the lr
for i in range(30):
    px = px - derivada(px) * lr
    py = my_function(px)
    plt.plot(px, py, 'b.')
    print(px)

plt.show()

# As we can see here, we are approximating to the minimum

# In a neuronal network, the x axis represents be the weights
# the y axis represents the error.