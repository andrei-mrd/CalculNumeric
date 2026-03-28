import numpy as np
from scipy import linalg

# ─────────────────────────────────────────────
# 1. Compute b_i = sum_j(s_j * a_ij) = A @ s
# ─────────────────────────────────────────────
def compute_b(A, s):
    return A @ s


# ─────────────────────────────────────────────
# 2. Householder QR decomposition
#    Returns R (in A), Q_tilde (= Q^T), and Q^T*b_init (in b)
# ─────────────────────────────────────────────
def householder_qr(A_in, b_in, epsilon=1e-10):
    n = A_in.shape[0]
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)
    Qt = np.eye(n)          # will become Q^T at the end

    for r in range(n - 1):
        # sigma = sum of squares of column r from row r downward
        sigma = np.dot(A[r:, r], A[r:, r])

        if sigma <= epsilon:
            print(f"  [Warning] Matrix nearly singular at step r={r + 1}")
            break

        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k

        beta = sigma - k * A[r, r]

        # Householder vector u (only indices r..n-1 are non-zero)
        u = np.zeros(n)
        u[r] = A[r, r] - k
        u[r + 1:] = A[r + 1:, r]

        # Transform columns j = r+1, ..., n-1  (A = P_r * A)
        for j in range(r + 1, n):
            gamma = np.dot(u[r:], A[r:, j]) / beta
            A[r:, j] -= gamma * u[r:]

        # Fix column r itself: a_rr = k, zeros below
        A[r, r] = k
        A[r + 1:, r] = 0.0

        # Transform b  (b = P_r * b)
        gamma = np.dot(u[r:], b[r:]) / beta
        b[r:] -= gamma * u[r:]

        # Transform Qt  (Qt = P_r * Qt)
        for j in range(n):
            gamma = np.dot(u[r:], Qt[r:, j]) / beta
            Qt[r:, j] -= gamma * u[r:]

    # Singularity check
    for i in range(n):
        if abs(A[i, i]) <= epsilon:
            raise ValueError(f"Singular matrix: |R[{i},{i}]| = {abs(A[i,i]):.2e} <= epsilon")

    return A, Qt, b   # A = R,  Qt = Q^T,  b = Q^T * b_init


# ─────────────────────────────────────────────
# Back substitution: solve upper-triangular Rx = b
# ─────────────────────────────────────────────
def back_substitution(R, b, n):
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(R[i, i + 1:], x[i + 1:])
        x[i] /= R[i, i]
    return x


# ─────────────────────────────────────────────
# 3. Solve Ax=b via Householder QR
# ─────────────────────────────────────────────
def solve_householder(A_init, b_init, epsilon=1e-10):
    R, Qt, Qtb = householder_qr(A_init, b_init, epsilon)
    x = back_substitution(R, Qtb, A_init.shape[0])
    return x, R, Qt


# ─────────────────────────────────────────────
# 5. Inverse via Householder QR
#    For each column j: solve Rx = (column j of Q^T)
# ─────────────────────────────────────────────
def inverse_householder(A_init, epsilon=1e-10):
    n = A_init.shape[0]
    R, Qt, _ = householder_qr(A_init, np.zeros(n), epsilon)
    A_inv = np.zeros((n, n))
    for j in range(n):
        b_j = Qt[:, j].copy()          # column j of Q^T
        A_inv[:, j] = back_substitution(R, b_j, n)
    return A_inv


# ─────────────────────────────────────────────
# Run all points for a given (n, A, s)
# ─────────────────────────────────────────────
def run(n, A, s, epsilon=1e-10, label=""):
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  n = {n}")
    print(f"  A =\n{A}")
    print(f"  s = {s}")

    # ── Point 1 ──────────────────────────────
    b = compute_b(A, s)
    print(f"\n[1] b = A @ s = {b}")

    A_init = A.copy()
    b_init = b.copy()

    # ── Point 3: Householder solve ───────────
    x_h, _, _ = solve_householder(A_init.copy(), b_init.copy(), epsilon)
    print(f"\n[3] x_Householder = {x_h}")

    # ── Point 3: Library QR solve ────────────
    Q_lib, R_lib = linalg.qr(A_init)
    x_qr = linalg.solve_triangular(R_lib, Q_lib.T @ b_init)
    print(f"[3] x_QR          = {x_qr}")
    print(f"[3] ||x_QR - x_Householder||₂ = {np.linalg.norm(x_qr - x_h):.2e}")

    # ── Point 4: Residual errors ─────────────
    e1 = np.linalg.norm(A_init @ x_h  - b_init)
    e2 = np.linalg.norm(A_init @ x_qr - b_init)
    e3 = np.linalg.norm(x_h  - s) / np.linalg.norm(s)
    e4 = np.linalg.norm(x_qr - s) / np.linalg.norm(s)
    print(f"\n[4] ||A·x_Householder - b||₂          = {e1:.2e}  (should be < 1e-6)")
    print(f"[4] ||A·x_QR - b||₂                   = {e2:.2e}  (should be < 1e-6)")
    print(f"[4] ||x_Householder - s||₂ / ||s||₂   = {e3:.2e}  (should be < 1e-6)")
    print(f"[4] ||x_QR - s||₂          / ||s||₂   = {e4:.2e}  (should be < 1e-6)")

    # ── Point 5: Inverse ─────────────────────
    A_inv_h   = inverse_householder(A_init.copy(), epsilon)
    A_inv_lib = np.linalg.inv(A_init)
    diff = np.linalg.norm(A_inv_h - A_inv_lib)
    print(f"\n[5] A_inv_Householder =\n{A_inv_h}")
    print(f"[5] ||A_inv_Householder - A_inv_lib|| = {diff:.2e}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    epsilon = 1e-10

    # ── Example from PDF ─────────────────────
    A_ex = np.array([[0, 0, 4],
                     [1, 2, 3],
                     [0, 1, 2]], dtype=float)
    s_ex = np.array([3, 2, 1], dtype=float)
    run(3, A_ex, s_ex, epsilon, label="Example from PDF")

    # ── Point 6: Random initialization ───────
    n_rand = 5
    np.random.seed(0)
    A_rand = np.random.rand(n_rand, n_rand) * 10
    s_rand = np.random.rand(n_rand) * 10
    run(n_rand, A_rand, s_rand, epsilon, label=f"Random n={n_rand}")
