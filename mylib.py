#Name-Eklavya Chauhan
#Roll No- 2311067
#My Library
#This library contains classes for complex numbers, random number generation, Gauss-Jordan elimination, LU decomposition using Crout's method.
import numpy as np

class MyComplex:
    def __init__(self, real, imag=0.0):
        self.r = real
        self.i = imag

    def display_cmplx(self):
        print(f"{self.r} + {self.i}j")

    def add_cmplx(self, c1, c2):
        self.r = c1.r + c2.r
        self.i = c1.i + c2.i
        return MyComplex(self.r, self.i)

    def sub_cmplx(self, c1, c2):
        self.r = c1.r - c2.r
        self.i = c1.i - c2.i
        return MyComplex(self.r, self.i)

    def mul_cmplx(self, c1, c2):
        self.r = c1.r * c2.r - c1.i * c2.i
        self.i = c1.i * c2.r + c1.r * c2.i
        return MyComplex(self.r, self.i)

    def mod_cmplx(self):
        return np.sqrt(self.r**2 + self.i**2)
class random_number:
    def __init__(self, a=1103515245, c=12345, m=32768, seed=1):
    
        #Linear Congruential Generator: X_{n+1} = (a * X_n + c) mod m

        #Parameters: a, c, m : LCG parameters, seed : initial seed value
       
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self):
        """Generate the next random number (integer)."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_float(self):
        """Generate the next random number as a float in [0, 1)."""
        return self.next() / self.m

    def generate_floats(self, n):
        """Generate a NumPy array of n floats in [0, 1)."""
        return np.array([self.next_float() for _ in range(n)])
class GaussJordan:
    def __init__(self, filename):
        # Read matrix from file
        self.A = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    row = list(map(float, line.split()))
                    self.A.append(row)
        self.n = len(self.A)

    def print_matrix(self, step=""): # Print the current state of the matrix
        if step:
            print(step)
        for row in self.A:
            print(row)
        print()

    def solve(self): # Perform Gauss-Jordan elimination
        n = self.n # Number of equations
        if n == 0:
            print("No equations to solve.")
            return []
        
        A = self.A  # Augmented matrix

        self.print_matrix("Initial Augmented Matrix:")

        for i in range(n):
            # Partial pivoting
            max_row = max(range(i, n), key=lambda r: abs(A[r][i])) # Find the row with the largest pivot element
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i] # Swap rows
                self.print_matrix(f"After swapping rows {i+1} and {max_row+1}:")
            # Normalize pivot row
            pivot = A[i][i]
            for k in range(len(A[i])): # Normalize the pivot row
                A[i][k] /= pivot 
            self.print_matrix(f"After normalizing rows{i+1}:") # Print the matrix after normalization
            # Eliminate other rows
            for j in range(n): # Eliminate the current column in all other rows
                if j != i:
                    factor = A[j][i]
                    for k in range(len(A[j])): # Subtract the pivot row from the current row
                        A[j][k] -= factor * A[i][k]
            self.print_matrix(f"After eliminating column{i+1}:") # Print the matrix after elimination

        # Extract solution
        solution = [A[i][-1] for i in range(n)]
        print("Final Reduced Row Echelon Form:")    # Print the final reduced row echelon form
        self.print_matrix()
        print("Solution Vector:", solution) # Extract the solution vector
        return solution
import numpy as np

class LUdecomposition: # Class for LU decomposition using Crout's method
    def __init__(self, filename):
        self.matrix = self._read_matrix(filename)
        self.A = self.crout_decomp(self.matrix) # Perform Crout's decomposition on the matrix
    def _read_matrix(self, filename):
        A = []
        with open(filename, "r") as f:
            for line in f:
                if line.strip():
                    row = list(map(float, line.split()))
                    A.append(row)
        return np.array(A)
    def crout_decomp(self, M):
        n = M.shape[0]
        L = [[0.0 for _ in range(n)] for _ in range(n)]
        U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # U with diagonal = 1
        for j in range(n):
            # Compute column j of L
            for i in range(j, n): # Lower triangle of L
                sum_l = sum(L[i][k] * U[k][j] for k in range(j))
                L[i][j] = M[i][j] - sum_l

            # Compute row j of U
            for i in range(j+1, n): # Upper triangle of U
                sum_u = sum(L[j][k] * U[k][i] for k in range(j))
                U[j][i] = (M[j][i] - sum_u) / L[j][j]
        # Store in a single matrix A
        A = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i >= j:   # Lower part (L)
                    A[i][j] = L[i][j]
                else:        # Upper part (U)
                    A[i][j] = U[i][j]
        return A
    def get_storage_matrix(self):
        #Return the storage matrix A (containing L and U).
        return self.A

