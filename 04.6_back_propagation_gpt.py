import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(output, target):
    return np.mean((output - target) ** 2) / 2

# Inicialización de pesos más pequeña
W = np.random.normal(scale=0.1, size=(2, 1))
print("Pesos iniciales:\n", W)

epochs = 3000  # Aumentar el número de épocas
lr = 0.1  # Reducir la tasa de aprendizaje
errors = []  # Lista para almacenar el error en cada época

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

    # Guardar el error de cada época
    epoch_error = mse(output, y)
    errors.append(epoch_error)

# Aplicar un umbral para obtener salidas binarias
output_binario = np.where(output >= 0.5, 1, 0)

# Resultados finales
for j in range(len(output_binario)):
    print(f"Entrada: {inputs[j]} - Salida esperada: {y[j]} - Predicción: {output_binario[j]}")

print("Pesos finales:\n", W)
print("Output final (binario):\n", output_binario)

# Graficar la evolución del error
plt.plot(errors)
plt.title("Evolución del Error durante el Entrenamiento")
plt.xlabel("Época")
plt.ylabel("Error MSE")
plt.show()
