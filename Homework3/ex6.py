"""
Exercise 6: Full program with random initialization for any dimension n.
Runs all exercises (1–5) with random data so the program works for any n.
"""

import numpy as np
import scipy.linalg

from ex2 import householder_qr, back_substitution


def run_all(n: int, epsilon: float = 1e-10, seed: int = None):
    rng = np.random.default_rng(seed)

    # Generate a random non-singular matrix A and vector s
    # Using random orthogonal base + random diagonal to ensure invertibility
    A = rng.standard_normal((n, n))
    # Make sure it is non-singular by checking rank; retry if needed
    while np.linalg.matrix_rank(A) < n:
        A = rng.standard_normal((n, n))

    s = rng.standard_normal(n)

    print(f"=== Running all exercises for n = {n}, epsilon = {epsilon:.0e} ===\n")

    # ------------------------------------------------------------------ #
    # Exercise 1: compute b = A @ s                                        #
    # ------------------------------------------------------------------ #
    b_init = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b_init[i] += s[j] * A[i][j]

    print("--- Exercise 1: b = A * s ---")
    if n <= 6:
        print(f"A =\n{A}")
        print(f"s = {s}")
        print(f"b = {b_init}")
    else:
        print(f"(n={n}, showing norms only)")
        print(f"||b||_2 = {np.linalg.norm(b_init):.6f}")

    # ------------------------------------------------------------------ #
    # Exercise 2: Householder QR                                           #
    # ------------------------------------------------------------------ #
    print("\n--- Exercise 2: Householder QR ---")
    R, Qt, b_transformed, singular = householder_qr(A, b_init, epsilon)

    if singular:
        print("Matrix A is singular — stopping.")
        return

    Q = Qt.T
    A_reconstructed = Q @ R
    print(f"||Q @ R - A||_2 = {np.linalg.norm(A_reconstructed - A):.6e}  (should be ~0)")
    print(f"||Q^T @ Q - I||_F = {np.linalg.norm(Qt @ Q - np.eye(n)):.6e}  (should be ~0)")

    # ------------------------------------------------------------------ #
    # Exercise 3: solve Ax = b                                             #
    # ------------------------------------------------------------------ #
    print("\n--- Exercise 3: Solve Ax = b ---")

    # Householder solution
    x_Householder = back_substitution(R, b_transformed, epsilon)

    # Library QR solution
    Q_lib, R_lib = scipy.linalg.qr(A)
    x_QR = scipy.linalg.solve_triangular(R_lib, Q_lib.T @ b_init)

    diff_x = np.linalg.norm(x_QR - x_Householder)
    print(f"||x_QR - x_Householder||_2 = {diff_x:.6e}")

    # ------------------------------------------------------------------ #
    # Exercise 4: error analysis                                           #
    # ------------------------------------------------------------------ #
    print("\n--- Exercise 4: Error analysis ---")

    err1 = np.linalg.norm(A @ x_Householder - b_init)
    err2 = np.linalg.norm(A @ x_QR - b_init)
    err3 = np.linalg.norm(x_Householder - s) / np.linalg.norm(s)
    err4 = np.linalg.norm(x_QR - s) / np.linalg.norm(s)

    print(f"||A @ x_Householder - b||_2          = {err1:.6e}")
    print(f"||A @ x_QR          - b||_2          = {err2:.6e}")
    print(f"||x_Householder - s||_2 / ||s||_2    = {err3:.6e}")
    print(f"||x_QR          - s||_2 / ||s||_2    = {err4:.6e}")

    threshold = 1e-6
    all_ok = all(e < threshold for e in [err1, err2, err3, err4])
    print(f"All errors < 1e-6: {all_ok}")

    # ------------------------------------------------------------------ #
    # Exercise 5: matrix inverse                                           #
    # ------------------------------------------------------------------ #
    print("\n--- Exercise 5: Matrix inverse ---")

    # Re-run Householder QR with dummy b (to get fresh R and Qt)
    dummy_b = np.zeros(n)
    R2, Qt2, _, _ = householder_qr(A, dummy_b, epsilon)

    A_inv_householder = np.zeros((n, n))
    for j in range(n):
        b_col = Qt2[:, j].copy()
        A_inv_householder[:, j] = back_substitution(R2, b_col, epsilon)

    A_inv_lib = np.linalg.inv(A)
    inv_diff = np.linalg.norm(A_inv_householder - A_inv_lib)
    print(f"||A^{{-1}}_Householder - A^{{-1}}_lib|| = {inv_diff:.6e}")
    print(f"||A @ A^{{-1}} - I||_F = {np.linalg.norm(A @ A_inv_householder - np.eye(n)):.6e}")


if __name__ == "__main__":
    import sys

    # Allow passing n as command-line argument: python ex6.py 10
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    # Fixed example from PDF (n=3)
    print("=" * 60)
    print("PDF example (n=3, fixed data)")
    print("=" * 60)

    A_pdf = np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]], dtype=float)
    s_pdf = np.array([3, 2, 1], dtype=float)
    b_pdf = A_pdf @ s_pdf

    R_pdf, Qt_pdf, b_tr_pdf, sing = householder_qr(A_pdf, b_pdf)
    if not sing:
        x_h = back_substitution(R_pdf, b_tr_pdf)
        Q_l, R_l = scipy.linalg.qr(A_pdf)
        x_q = scipy.linalg.solve_triangular(R_l, Q_l.T @ b_pdf)
        print(f"x_Householder = {x_h}  (expected: {s_pdf})")
        print(f"x_QR          = {x_q}")
        err = np.linalg.norm(x_h - s_pdf) / np.linalg.norm(s_pdf)
        print(f"Relative error (Householder) = {err:.6e}\n")

    # Random example with user-supplied (or default) n
    print("=" * 60)
    print(f"Random initialization (n={n})")
    print("=" * 60)
    run_all(n=n, epsilon=1e-10, seed=42)
