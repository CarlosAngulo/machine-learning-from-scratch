import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(output, target):
    return np.mean((output - target) ** 2) / 2

# Inicialización de pesos
W = np.random.normal(scale=0.5, size=(2, 1))
print("Pesos iniciales:\n", W)

epochs = 10000
lr = 0.05

# Entrenamiento
for i in range(epochs):
    # Forward
    z = np.dot(inputs, W)
    output = sigmoid(z)

    # Backpropagation
    error = output - y
    delta = error * (output * (1 - output))  # Ajuste al gradiente
    derivada = np.dot(inputs.T, delta)
    
    # Gradient descent
    W -= derivada * lr

# Resultados finales
for j in range(len(output)):
    print(f"Entrada: {inputs[j]} - Salida esperada: {y[j]} - Predicción: {round(output[j, 0])}")

print("Pesos finales:\n", W)
print("Output final:\n", output)
