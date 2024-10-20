import matplotlib.pyplot as plt
import numpy as np

def my_function(x):
    return x**2 - 5*x + 2

def derivada(x):
    return 2*x - 5

x = np.linspace(-20, 20, 100)
y = my_function(x)

plt.plot(x, y)

y1 = derivada(-10)

init_x = 15
init_y = my_function(init_x)

dx = derivada(init_x)
# define the leanring rate which multiplies the dx
lr = 0.1
px = init_x - dx  * lr
py = my_function(px)

plt.plot(init_x, init_y, 'r.')
plt.plot(px, py, 'r.')
plt.show()

# As we can see here, the step are smaller than initially, but still pointing to the gradient