import numpy as np
from scipy import linalg

# ─────────────────────────────────────────────
# TASK 1: Compute b_i = sum_j( s_j * a_ij )
# ─────────────────────────────────────────────
def compute_b(A, s):
    n = len(s)
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * A[i][j]
    return b

# ─────────────────────────────────────────────
# TASK 2: QR decomposition via Householder
# ─────────────────────────────────────────────
def householder_qr(A, eps):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for k in range(n):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < eps:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x
        v = v / np.linalg.norm(v)

        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    R[np.abs(R) < eps] = 0.0
    return Q, R

# ─────────────────────────────────────────────
# TASK 3: Solve Ax = b using both QR methods
# ─────────────────────────────────────────────
def solve_qr_library(A, b):
    """Solve Ax = b using scipy's built-in QR decomposition."""
    Q, R = linalg.qr(A)
    # Ax = b → QRx = b → Rx = Q^T b
    Qt_b = Q.T @ b
    x = linalg.solve_triangular(R, Qt_b)
    return x

def solve_qr_householder(Q, R, b, eps):
    """Solve Ax = b using our Householder QR: Rx = Q^T b."""
    Qt_b = Q.T @ b
    n = R.shape[0]

    # Back substitution to solve Rx = Qt_b
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < eps:
            raise ValueError(f"Matricea este singulara la diagonala [{i}][{i}]")
        x[i] = Qt_b[i]
        for j in range(i + 1, n):
            x[i] -= R[i][j] * x[j]
        x[i] /= R[i][i]
    return x

# ─────────────────────────────────────────────
# Helper: print matrix/vector
# ─────────────────────────────────────────────
def print_matrix(name, M, eps):
    print(f"\nMatricea {name}:")
    if M.ndim == 1:
        for i, val in enumerate(M):
            v = 0.0 if abs(val) < eps else val
            print(f"  [{i+1}] = {v:10.6f}")
    else:
        for row in M:
            print("  " + "  ".join(
                f"{(0.0 if abs(v) < eps else v):10.4f}" for v in row
            ))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    n   = int(input("n = "))
    eps = float(input("eps = "))

    # --- Read matrix A ---
    print(f"\nIntroduceti matricea A ({n}x{n}):")
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            val = float(input(f"  A[{i+1}][{j+1}] = "))
            row.append(val)
        A.append(row)
    A = np.array(A, dtype=float)

    # --- Read vector s (for task 1) ---
    print(f"\nIntroduceti vectorul s ({n} elemente):")
    s = [float(input(f"  s[{j+1}] = ")) for j in range(n)]
    s = np.array(s, dtype=float)

    # ── TASK 1 ──────────────────────────────
    print("\n" + "="*50)
    print("TASK 1: Calculul vectorului b")
    print("="*50)
    b = compute_b(A, s)
    print_matrix("b", b, eps)

    # ── TASK 2 ──────────────────────────────
    print("\n" + "="*50)
    print("TASK 2: Descompunerea QR (Householder)")
    print("="*50)
    Q_house, R_house = householder_qr(A, eps)
    print_matrix("Q (Householder)", Q_house, eps)
    print_matrix("R (Householder)", R_house, eps)

    recon_err = np.linalg.norm(A - Q_house @ R_house)
    orth_err  = np.linalg.norm(Q_house.T @ Q_house - np.eye(n))
    print(f"\n  ||A - Q*R||         = {recon_err:.2e}")
    print(f"  ||Q^T*Q - I||       = {orth_err:.2e}")

    # ── TASK 3 ──────────────────────────────
    print("\n" + "="*50)
    print("TASK 3: Rezolvarea sistemului Ax = b")
    print("="*50)

    # Solution via scipy QR
    x_qr = solve_qr_library(A, b)
    print_matrix("x_QR (scipy)", x_qr, eps)

    # Solution via our Householder QR
    x_house = solve_qr_householder(Q_house, R_house, b, eps)
    print_matrix("x_Householder", x_house, eps)

    # Difference between the two solutions
    diff = np.linalg.norm(x_qr - x_house)
    print(f"\n  ||x_QR - x_Householder||_2 = {diff:.2e}")

    # Verify both solutions
    print(f"\n  ||A*x_QR - b||_2          = {np.linalg.norm(A @ x_qr - b):.2e}")
    print(f"  ||A*x_Householder - b||_2 = {np.linalg.norm(A @ x_house - b):.2e}")

if __name__ == "__main__":
    main()