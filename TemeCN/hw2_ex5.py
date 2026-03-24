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
    norm_s    = np.linalg.norm(s)
    res_house = np.linalg.norm(A_init @ x_house - b_init)
    res_qr    = np.linalg.norm(A_init @ x_qr    - b_init)
    rel_house = np.linalg.norm(x_house - s) / norm_s
    rel_qr    = np.linalg.norm(x_qr    - s) / norm_s
    return res_house, res_qr, rel_house, rel_qr

# ─────────────────────────────────────────────
# TASK 5: Compute A^-1 via Householder QR
#         and compare with numpy/scipy inverse
# ─────────────────────────────────────────────
def invert_via_householder_qr(A, Q, R, eps):
    """
    A^-1 = R^-1 * Q^T
    We compute R^-1 by solving R * col = e_i for each standard basis vector e_i,
    then A^-1 = R^-1 * Q^T
    """
    n = A.shape[0]
    R_inv = np.zeros((n, n))

    # Solve R * x = e_i for each column i (back substitution)
    for col in range(n):
        e = np.zeros(n)
        e[col] = 1.0
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(R[i][i]) < eps:
                raise ValueError(f"Matricea este singulara: R[{i}][{i}] ~ 0")
            x[i] = e[i]
            for j in range(i + 1, n):
                x[i] -= R[i][j] * x[j]
            x[i] /= R[i][i]
        R_inv[:, col] = x

    # A^-1 = R^-1 * Q^T
    A_inv_house = R_inv @ Q.T
    return A_inv_house

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

def print_errors(res_house, res_qr, rel_house, rel_qr):
    threshold = 1e-6
    print("\n" + "="*60)
    print("TASK 4: Erori de calcul")
    print("="*60)

    def check(val):
        return "✓ < 1e-6" if val < threshold else "✗ >= 1e-6"

    print(f"\n  ||A * x_Householder - b||_2        = {res_house:.6e}  {check(res_house)}")
    print(f"  ||A * x_QR          - b||_2        = {res_qr:.6e}  {check(res_qr)}")
    print(f"\n  ||x_Householder - s||_2 / ||s||_2  = {rel_house:.6e}  {check(rel_house)}")
    print(f"  ||x_QR          - s||_2 / ||s||_2  = {rel_qr:.6e}  {check(rel_qr)}")

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
    A_init = A.copy()

    # --- Read vector s ---
    print(f"\nIntroduceti vectorul s ({n} elemente):")
    s = np.array([float(input(f"  s[{j+1}] = ")) for j in range(n)])

    # ── TASK 1 ──────────────────────────────
    print("\n" + "="*60)
    print("TASK 1: Calculul vectorului b")
    print("="*60)
    b = compute_b(A, s)
    b_init = b.copy()
    print_matrix("b = A * s", b, eps)

    # ── TASK 2 ──────────────────────────────
    print("\n" + "="*60)
    print("TASK 2: Descompunerea QR (Householder)")
    print("="*60)
    Q_house, R_house = householder_qr(A_init, eps)
    print_matrix("Q (Householder)", Q_house, eps)
    print_matrix("R (Householder)", R_house, eps)
    print(f"\n  ||A - Q*R||   = {np.linalg.norm(A_init - Q_house @ R_house):.2e}")
    print(f"  ||Q^T*Q - I|| = {np.linalg.norm(Q_house.T @ Q_house - np.eye(n)):.2e}")

    # ── TASK 3 ──────────────────────────────
    print("\n" + "="*60)
    print("TASK 3: Rezolvarea sistemului Ax = b")
    print("="*60)
    x_qr    = solve_qr_library(A_init, b_init)
    x_house = solve_qr_householder(Q_house, R_house, b_init, eps)
    print_matrix("x_QR (scipy)",  x_qr,    eps)
    print_matrix("x_Householder", x_house, eps)
    print(f"\n  ||x_QR - x_Householder||_2 = {np.linalg.norm(x_qr - x_house):.2e}")

    # ── TASK 4 ──────────────────────────────
    res_house, res_qr, rel_house, rel_qr = compute_errors(
        A_init, b_init, s, x_house, x_qr
    )
    print_errors(res_house, res_qr, rel_house, rel_qr)

    # ── TASK 5 ──────────────────────────────
    print("\n" + "="*60)
    print("TASK 5: Inversa matricei A")
    print("="*60)

    # Our Householder-based inverse: A^-1 = R^-1 * Q^T
    A_inv_house = invert_via_householder_qr(A_init, Q_house, R_house, eps)

    # Library inverse (numpy)
    A_inv_bibl  = np.linalg.inv(A_init)

    print_matrix("A^-1 (Householder)", A_inv_house, eps)
    print_matrix("A^-1 (numpy)",       A_inv_bibl,  eps)

    # Compare the two inverses
    diff_norm = np.linalg.norm(A_inv_house - A_inv_bibl)
    print(f"\n  ||A_inv_Householder - A_inv_bibl|| = {diff_norm:.6e}", end="  ")
    print("✓ < 1e-6" if diff_norm < 1e-6 else "✗ >= 1e-6")

    # Extra sanity checks
    print(f"\n  ||A * A_inv_Householder - I||      = {np.linalg.norm(A_init @ A_inv_house - np.eye(n)):.6e}")
    print(f"  ||A * A_inv_bibl        - I||      = {np.linalg.norm(A_init @ A_inv_bibl  - np.eye(n)):.6e}")

if __name__ == "__main__":
    main()