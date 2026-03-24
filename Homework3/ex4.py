"""
Exercise 4: Compute and display the following errors (all should be < 1e-6):

  ||A_init @ x_Householder - b_init||_2
  ||A_init @ x_QR          - b_init||_2
  ||x_Householder - s||_2 / ||s||_2
  ||x_QR          - s||_2 / ||s||_2
"""

import numpy as np
import scipy.linalg

from ex2 import householder_qr, back_substitution

# --- Data from PDF example ---
n = 3
epsilon = 1e-10

A_init = np.array([
    [0, 0, 4],
    [1, 2, 3],
    [0, 1, 2]
], dtype=float)

s = np.array([3, 2, 1], dtype=float)
b_init = A_init @ s   # from exercise 1

print("=== Exercise 4: Error analysis ===")
print(f"A =\n{A_init}")
print(f"s = {s}")
print(f"b = {b_init}")

# ------------------------------------------------------------------ #
# Householder solution                                                 #
# ------------------------------------------------------------------ #
R_h, Qt_h, b_h, singular = householder_qr(A_init, b_init, epsilon)

if singular:
    print("Matrix A is singular!")
else:
    x_Householder = back_substitution(R_h, b_h, epsilon)

    # ------------------------------------------------------------------ #
    # Library QR solution                                                  #
    # ------------------------------------------------------------------ #
    Q_lib, R_lib = scipy.linalg.qr(A_init)
    x_QR = scipy.linalg.solve_triangular(R_lib, Q_lib.T @ b_init)

    # ------------------------------------------------------------------ #
    # Error computations                                                   #
    # ------------------------------------------------------------------ #
    err1 = np.linalg.norm(A_init @ x_Householder - b_init)
    err2 = np.linalg.norm(A_init @ x_QR - b_init)
    err3 = np.linalg.norm(x_Householder - s) / np.linalg.norm(s)
    err4 = np.linalg.norm(x_QR - s) / np.linalg.norm(s)

    print(f"\n||A_init @ x_Householder - b_init||_2          = {err1:.6e}")
    print(f"||A_init @ x_QR          - b_init||_2          = {err2:.6e}")
    print(f"||x_Householder - s||_2 / ||s||_2              = {err3:.6e}")
    print(f"||x_QR          - s||_2 / ||s||_2              = {err4:.6e}")

    threshold = 1e-6
    print(f"\nAll errors < {threshold}: "
          f"{all(e < threshold for e in [err1, err2, err3, err4])}")
