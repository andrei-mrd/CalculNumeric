"""
Exercise 1: Compute vector b from matrix A and vector s.
b_i = sum_{j=1}^{n} s_j * a_{ij},  i = 1,...,n
This is equivalent to b = A @ s (matrix-vector product).
"""

import numpy as np

# --- Data from PDF example ---
n = 3
epsilon = 1e-10

A = np.array([
    [0, 0, 4],
    [1, 2, 3],
    [0, 1, 2]
], dtype=float)

s = np.array([3, 2, 1], dtype=float)

# b_i = sum_{j=1}^{n} s_j * a_{ij}
b = np.zeros(n)
for i in range(n):
    for j in range(n):
        b[i] += s[j] * A[i][j]

print("=== Exercise 1: Compute b = A * s ===")
print(f"A =\n{A}")
print(f"s = {s}")
print(f"b = {b}")
print(f"Verification with numpy: b = {A @ s}")
