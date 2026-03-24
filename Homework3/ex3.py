"""
Exercise 3: Solve the linear system Ax = b using:
  1. Library QR decomposition  -> x_QR
  2. Householder QR (exercise 2) -> x_Householder

Display: ||x_QR - x_Householder||_2
"""

import numpy as np
import scipy.linalg

from ex2 import householder_qr, back_substitution

# --- Random data ---
n = 5
epsilon = 1e-10
rng = np.random.default_rng()

A = rng.integers(0, 10, size=(n, n)).astype(float)
s = rng.integers(1, 10, size=n).astype(float)
b = A @ s   # from exercise 1

print("=== Exercise 3: Solve Ax = b ===")
print(f"A =\n{A}")
print(f"b = {b}")

# ------------------------------------------------------------------ #
# Method 1: Library QR (scipy)                                        #
# ------------------------------------------------------------------ #
Q_lib, R_lib = scipy.linalg.qr(A)
# Ax = b  <=>  QRx = b  <=>  Rx = Q^T @ b
b_lib = Q_lib.T @ b
x_QR = scipy.linalg.solve_triangular(R_lib, b_lib)

print(f"\nx_QR (library)        = {x_QR}")

# ------------------------------------------------------------------ #
# Method 2: Householder QR (exercise 2)                               #
# ------------------------------------------------------------------ #
R_h, Qt_h, b_h, singular = householder_qr(A, b, epsilon)

if singular:
    print("Matrix A is singular — cannot solve!")
else:
    x_Householder = back_substitution(R_h, b_h, epsilon)
    print(f"x_Householder         = {x_Householder}")
    print(f"\n||x_QR - x_Householder||_2 = {np.linalg.norm(x_QR - x_Householder):.6e}")
