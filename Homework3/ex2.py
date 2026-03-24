"""
Exercise 2: QR decomposition of matrix A using Householder algorithm.

Algorithm:
- Q_tilde = I_n  (at the end Q_tilde = Q^T)
- for r = 1,...,n-1:
    sigma = sum(a[i][r]^2, i=r..n)
    if sigma <= eps: break (A singular)
    k = sqrt(sigma)
    if a[r][r] > 0: k = -k
    beta = sigma - k * a[r][r]
    u[r] = a[r][r] - k;  u[i] = a[i][r] for i=r+1,...,n
    // A = P_r * A
    for j = r+1,...,n:
        gamma = (sum u[i]*a[i][j], i=r..n) / beta
        for i = r,...,n: a[i][j] -= gamma * u[i]
    a[r][r] = k;  a[i][r] = 0 for i=r+1,...,n
    // b = P_r * b
    gamma = (sum u[i]*b[i], i=r..n) / beta
    for i = r,...,n: b[i] -= gamma * u[i]
    // Q_tilde = P_r * Q_tilde
    for j = 1,...,n:
        gamma = (sum u[i]*q[i][j], i=r..n) / beta
        for i = r,...,n: q[i][j] -= gamma * u[i]

At the end: R is stored in A, Q^T is in Q_tilde, b holds Q^T * b_init.
"""

import numpy as np
import math


def householder_qr(A_input, b_input, epsilon=1e-10):
    """
    Compute QR decomposition of A via Householder reflections.

    Parameters
    ----------
    A_input : ndarray (n, n)
    b_input : ndarray (n,)
    epsilon  : float, precision threshold

    Returns
    -------
    R        : ndarray (n, n), upper triangular (R = modified A in-place)
    Qt       : ndarray (n, n), Q^T
    b_trans  : ndarray (n,),  Q^T @ b_init
    singular : bool, True if A is detected singular
    """
    n = A_input.shape[0]
    A = A_input.copy().astype(float)
    b = b_input.copy().astype(float)
    Qt = np.eye(n)          # will accumulate Q^T

    for r in range(n - 1):  # r = 0,...,n-2  (1-indexed: 1,...,n-1)
        # --- Build P_r: compute sigma, k, beta, u ---
        sigma = sum(A[i][r] ** 2 for i in range(r, n))

        if sigma <= epsilon:
            return A, Qt, b, True   # singular matrix

        k = math.sqrt(sigma)
        if A[r][r] > 0:
            k = -k

        beta = sigma - k * A[r][r]

        u = np.zeros(n)
        u[r] = A[r][r] - k
        for i in range(r + 1, n):
            u[i] = A[i][r]

        # --- A = P_r * A: transform columns j = r+1,...,n-1 ---
        for j in range(r + 1, n):
            gamma = sum(u[i] * A[i][j] for i in range(r, n)) / beta
            for i in range(r, n):
                A[i][j] -= gamma * u[i]

        # --- Transform column r of A (set to triangular form) ---
        A[r][r] = k
        for i in range(r + 1, n):
            A[i][r] = 0.0

        # --- b = P_r * b ---
        gamma = sum(u[i] * b[i] for i in range(r, n)) / beta
        for i in range(r, n):
            b[i] -= gamma * u[i]

        # --- Qt = P_r * Qt: transform all columns of Qt ---
        for j in range(n):
            gamma = sum(u[i] * Qt[i][j] for i in range(r, n)) / beta
            for i in range(r, n):
                Qt[i][j] -= gamma * u[i]

    return A, Qt, b, False   # A is now R


def back_substitution(R, b, epsilon=1e-10):
    """
    Solve upper-triangular system R @ x = b by back substitution.

    Parameters
    ----------
    R : ndarray (n, n), upper triangular
    b : ndarray (n,)
    epsilon : float, threshold for singularity check

    Returns
    -------
    x : ndarray (n,)
    """
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < epsilon:
            raise ValueError(f"Singular matrix: R[{i}][{i}] = {R[i][i]}")
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i][j] * x[j]
        x[i] /= R[i][i]
    return x


# ------------------------------------------------------------------ #
#  Demo with the PDF example                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    rng = np.random.default_rng()
    n = 10
    A = rng.integers(0, 10, size=(n, n)).astype(float)
    s = rng.integers(1, 10, size=n).astype(float)
    b = A @ s   # b from exercise 1

    print("=== Exercise 2: Householder QR decomposition ===")
    print(f"A =\n{A}")
    print(f"b = {b}")

    A_init = A.copy()
    R, Qt, b_trans, singular = householder_qr(A, b)

    if singular:
        print("Matrix A is singular!")
    else:
        Q = Qt.T
        print(f"\nR =\n{R}")
        print(f"\nQ =\n{Q}")
        print(f"\nQ^T =\n{Qt}")
        print(f"\nb_transformed (Q^T @ b) = {b_trans}")

        print(f"\nVerification ||Q @ R - A_init||_2 = {np.linalg.norm(Q @ R - A_init):.2e}")
        print(f"Verification Q^T @ Q = I:\n{np.round(Qt @ Q, 10)}")
