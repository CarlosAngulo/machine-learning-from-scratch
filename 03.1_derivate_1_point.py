import matplotlib.pyplot as plt
import numpy as np

def my_function(x):
    return x**2 - 5*x + 2

def derivada(x):
    return 2*x - 5

# This method creates a linear function
# based on 1 point and its gradient (m, derivated)
def line_on_dot(x, x1, y, m):
    return m * (x - x1) + y 

def plot(init_x):
    x = np.linspace(-20, 20, 100)
    y = my_function(x)
    plt.plot(x, y)
    
    init_y = my_function(init_x)
    plt.plot(init_x, init_y, 'r.')
    
    m = derivada(init_x)
    plt.plot(x, line_on_dot(x, init_x, init_y, m))
    plt.show()


plot(-10.5)
