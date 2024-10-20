import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])



A = np.array([[1, 2, 3, 7], [4, 5, 6, 8]]) #2 Rows, 4 Columns
B = np.array([[1,2,1], [3,4,1], [5,6,1], [3,2,1]]) #4 Rows, 3 Columns
C = np.array([[1,2], [3,4], [5,6]]) #4 Rows, 3 Columns

#The product AxB is possible because 
# A Columns = B Rows

# print(np.dot(A, B))

def multiplicar(M1, M2):
    try:
        result = np.dot(M1, M2)
    except:
        result = 'Número de columnas de primera matriz debe ser igual a filas de segunda matriz'
    return result

print("1 =>", multiplicar([[1,2,1], [3,4,1]], B))
# Mostrará el error especifiado

print("2 =>", multiplicar(A, B))
#Result: 
#[[43 42 13]
#[73 80 23]]
# Rows: Number of Rows in A and Number of Columns in B => 2 Rows, 3 Columns

# TRANSPOSITION
print("3 =>", np.dot(C, A).T)
# We are transponding each vector to coincide rows and columns