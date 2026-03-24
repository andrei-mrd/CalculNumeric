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
    Q, R = linalg.qr(A)
    Qt_b = Q.T @ b
    return linalg.solve_triangular(R, Qt_b)

def solve_qr_householder(Q, R, b, eps):
    Qt_b = Q.T @ b
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < eps:
            raise ValueError(f"Matricea este singulara la [{i}][{i}]")
        x[i] = Qt_b[i]
        for j in range(i + 1, n):
            x[i] -= R[i][j] * x[j]
        x[i] /= R[i][i]
    return x

# ─────────────────────────────────────────────
# TASK 4: Compute and display all errors
# ─────────────────────────────────────────────
def compute_errors(A_init, b_init, s, x_house, x_qr):
    norm_s = np.linalg.norm(s)

    # Residual errors: how well does the solution satisfy Ax = b
    res_house = np.linalg.norm(A_init @ x_house - b_init)
    res_qr    = np.linalg.norm(A_init @ x_qr    - b_init)

    # Relative errors: how close are solutions to the true solution s
    rel_house = np.linalg.norm(x_house - s) / norm_s
    rel_qr    = np.linalg.norm(x_qr    - s) / norm_s

    return res_house, res_qr, rel_house, rel_qr

def print_errors(res_house, res_qr, rel_house, rel_qr):
    threshold = 1e-6
    print("\n" + "="*55)
    print("TASK 4: Erori de calcul")
    print("="*55)

    def check(val):
        return "✓  < 1e-6" if val < threshold else "✗  >= 1e-6"

    print(f"\n  ||A_init * x_Householder - b_init||_2  = {res_house:.6e}  {check(res_house)}")
    print(f"  ||A_init * x_QR          - b_init||_2  = {res_qr:.6e}  {check(res_qr)}")
    print(f"\n  ||x_Householder - s||_2 / ||s||_2      = {rel_house:.6e}  {check(rel_house)}")
    print(f"  ||x_QR          - s||_2 / ||s||_2      = {rel_qr:.6e}  {check(rel_qr)}")
    print()

# ─────────────────────────────────────────────
# Helper: print matrix/vector
# ─────────────────────────────────────────────
def print_matrix(name, M, eps):
    print(f"\n  {name}:")
    if M.ndim == 1:
        for i, val in enumerate(M):
            v = 0.0 if abs(val) < eps else val
            print(f"    [{i+1}] = {v:12.6f}")
    else:
        for row in M:
            print("    " + "  ".join(
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
    A_init = A.copy()  # save original A

    # --- Read vector s ---
    print(f"\nIntroduceti vectorul s ({n} elemente):")
    s = np.array([float(input(f"  s[{j+1}] = ")) for j in range(n)])

    # ── TASK 1 ──────────────────────────────
    print("\n" + "="*55)
    print("TASK 1: Calculul vectorului b")
    print("="*55)
    b = compute_b(A, s)
    b_init = b.copy()  # save original b
    print_matrix("b = A * s", b, eps)

    # ── TASK 2 ──────────────────────────────
    print("\n" + "="*55)
    print("TASK 2: Descompunerea QR (Householder)")
    print("="*55)
    Q_house, R_house = householder_qr(A, eps)
    print_matrix("Q (Householder)", Q_house, eps)
    print_matrix("R (Householder)", R_house, eps)
    print(f"\n  ||A - Q*R||   = {np.linalg.norm(A_init - Q_house @ R_house):.2e}")
    print(f"  ||Q^T*Q - I|| = {np.linalg.norm(Q_house.T @ Q_house - np.eye(n)):.2e}")

    # ── TASK 3 ──────────────────────────────
    print("\n" + "="*55)
    print("TASK 3: Rezolvarea sistemului Ax = b")
    print("="*55)
    x_qr    = solve_qr_library(A_init, b_init)
    x_house = solve_qr_householder(Q_house, R_house, b_init, eps)
    print_matrix("x_QR (scipy)",    x_qr,    eps)
    print_matrix("x_Householder",   x_house, eps)
    print(f"\n  ||x_QR - x_Householder||_2 = {np.linalg.norm(x_qr - x_house):.2e}")

    # ── TASK 4 ──────────────────────────────
    res_house, res_qr, rel_house, rel_qr = compute_errors(
        A_init, b_init, s, x_house, x_qr
    )
    print_errors(res_house, res_qr, rel_house, rel_qr)

if __name__ == "__main__":
    main()