#Name- Eklavya Chauhan
#Roll no-2311067

#Ques 1

import numpy as np

def read_matrix(filename):
    A = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                row = list(map(float, line.split()))
                A.append(row)
    return np.array(A)
def crout_decomp(A):
    n = len(A)
    L = [[0.0 for w in range(n)] for w in range(n)]
    U = np.identity(n)  # U with diagonal = 1

    for j in range(n):
        # Compute column j of L
        for i in range(j, n):
            sum_l = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = A[i][j] - sum_l

        # Compute row j of U
        for i in range(j+1, n):
            sum_u = sum(L[j][k] * U[k][i] for k in range(j))
            U[j][i] = (A[j][i] - sum_u) / L[j][j]

    return L, U
def store_LU(L, U):
    # Store Crout decomposition in one storage matrix A
    n = len(L)
    A = [[0.0 for e in range(n)] for e in range(n)]

    for i in range(n):
        for j in range(n):
            if i >= j:      # lower triangle (L)
                A[i][j] = L[i][j]
            else:           # upper triangle (U)
                A[i][j] = U[i][j]

    return A
def multiply_matrices(L, U):
    n = len(L)
    product = [[0.0 for u in range(n)] for u in range(n)] # Initialize product matrix
    # Perform matrix multiplication
    for i in range(n):
        for j in range(n):
            for k in range(n):
                product[i][j] += L[i][k] * U[k][j]
    return product
filename = "matrix1.txt" # Read input matrix from file
matrixA = read_matrix(filename)
print("Input Matrix")
print(matrixA)

# Perform LU decomposition (Crout's method)
L, U = crout_decomp(matrixA.copy())

print("Crout LU Decomposition")

# Store LU in a single matrix
A = store_LU(L, U)
print("Stored LU Matrix (A)")
print(A)

# Multiply L and U manually
product = multiply_matrices(L, U)

print("Verification")
print("Reconstructed Matrix (L * U):")
for row in product:
    print(row)

# Verification
if np.allclose(matrixA, product):
    print("Verification successful: L * U matches the original matrix.")
else:
    print("Verification failed: L * U does not match the original matrix.")

#Ques 2

from mylib import LUdecomposition   # Importing LUdecomposition class from mylib.py

def extract_LU(A): # Extract L and U from the compact storage matrix A (Crout).
    n = len(A)
    L = np.zeros((n, n))
    U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # U with diagonal = 1
    # Fill L and U from A
    for i in range(n):
        for j in range(n):
            if i >= j: # Lower part (L)
                L[i][j] = A[i][j]
            else: # Upper part (U)
                U[i][j] = A[i][j]
    return L, U

def forward_substitution(L, b):
    # Solve L y = b
    n = len(b)
    y = [0.0 for _ in range(n)]
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i)) # Sum of products for forward substitution
        y[i] = (b[i] - s) / L[i][i]
    return y

def backward_substitution(U, y):
    #Solve U x = y.
    n = len(y)
    x = [0.0 for _ in range(n)]
    for i in reversed(range(n)):
        s = sum(U[i][j] * x[j] for j in range(i+1, n)) # Sum of products for backward substitution
        x[i] = (y[i] - s) / U[i][i]
    return x

# Read augmented matrix [A|b]
augmented = []
with open("matrix2.txt", "r") as f: # Read augmented matrix from file
    for line in f:
        if line.strip():
            augmented.append([float(val) for val in line.split()])

# Split coefficient matrix and RHS
coeff = [row[:-1] for row in augmented]
b = [row[-1] for row in augmented]

# Perform LU decomposition
with open("coeff.txt", "w") as f: # Write coefficient matrix to file
    # We want to pass only coeff file to the LUdecomposition class  
    for row in coeff:
        f.write(" ".join(str(val) for val in row) + "\n")

lu = LUdecomposition("coeff.txt")
A = lu.get_storage_matrix() # Get the storage matrix A (containing L and U).

L, U = extract_LU(A) # Extract L and U from the storage matrix

y = forward_substitution(L, b) # Solve L y = b
x = backward_substitution(U, y) # Solve U x = y

print("Solution vector x:")
print(x)
