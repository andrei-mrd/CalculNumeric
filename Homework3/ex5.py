"""
Exercise 5: Compute the inverse of A using the Householder QR decomposition
(exercise 2) and compare with the library inverse.

Procedure:
  - Compute QR decomposition once (get R and Q^T).
  - For each column j = 1,...,n:
      b = Q^T @ e_j  =  j-th column of Q^T  =  Qt[:, j]
      Solve R @ x = b  (back substitution)
      x is column j of A^{-1}
  - Display: ||A_Householder^{-1} - A_bibl^{-1}||
"""

import numpy as np
import scipy.linalg

from ex2 import householder_qr, back_substitution

# --- Random data ---
n = 5
epsilon = 1e-10
rng = np.random.default_rng()

A_init = rng.integers(0, 10, size=(n, n)).astype(float)
s = rng.integers(1, 10, size=n).astype(float)
b_init = A_init @ s   # from exercise 1

print("=== Exercise 5: Matrix inverse via Householder QR ===")

# ------------------------------------------------------------------ #
# Householder QR (we pass a dummy b; the real b is not needed here)   #
# ------------------------------------------------------------------ #
dummy_b = np.zeros(n)
R, Qt, _, singular = householder_qr(A_init, dummy_b, epsilon)

if singular:
    print("Matrix A is singular — inverse does not exist!")
else:
    A_inv_householder = np.zeros((n, n))

    for j in range(n):
        # b = Q^T @ e_j = j-th column of Q^T = Qt[:, j]
        b_col = Qt[:, j].copy()

        # Solve R @ x = b_col
        x_col = back_substitution(R, b_col, epsilon)

        # Store as column j of the inverse
        A_inv_householder[:, j] = x_col

    # ------------------------------------------------------------------ #
    # Library inverse                                                      #
    # ------------------------------------------------------------------ #
    A_inv_lib = np.linalg.inv(A_init)

    diff_norm = np.linalg.norm(A_inv_householder - A_inv_lib)

    print(f"\nA_init =\n{A_init}")
    print(f"\nA^{{-1}} Householder =\n{A_inv_householder}")
    print(f"\nA^{{-1}} library     =\n{A_inv_lib}")
    print(f"\n||A_Householder^{{-1}} - A_bibl^{{-1}}|| = {diff_norm:.6e}")

    # Sanity check: A @ A^{-1} should be close to I
    print(f"\nVerification A @ A^{{-1}} =\n{np.round(A_init @ A_inv_householder, 10)}")
