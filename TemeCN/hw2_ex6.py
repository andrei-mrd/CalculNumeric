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
    return linalg.solve_triangular(R, Q.T @ b)

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
# TASK 4: Compute errors
# ─────────────────────────────────────────────
def compute_errors(A, b, s, x_house, x_qr):
    norm_s    = np.linalg.norm(s)
    res_house = np.linalg.norm(A @ x_house - b)
    res_qr    = np.linalg.norm(A @ x_qr    - b)
    rel_house = np.linalg.norm(x_house - s) / norm_s
    rel_qr    = np.linalg.norm(x_qr    - s) / norm_s
    return res_house, res_qr, rel_house, rel_qr

# ─────────────────────────────────────────────
# TASK 5: Invert A via Householder QR
# ─────────────────────────────────────────────
def invert_via_householder_qr(A, Q, R, eps):
    n = A.shape[0]
    R_inv = np.zeros((n, n))
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
    return R_inv @ Q.T

# ─────────────────────────────────────────────
# TASK 6: Random data generation
# ─────────────────────────────────────────────
def generate_random_data(n, seed=None, value_range=(-10.0, 10.0)):
    """
    Generate a random invertible matrix A and vector s.
    Ensures A is invertible by checking condition number.
    """
    rng = np.random.default_rng(seed)
    low, high = value_range

    max_attempts = 100
    for attempt in range(max_attempts):
        A = rng.uniform(low, high, size=(n, n))
        cond = np.linalg.cond(A)
        if cond < 1e10:  # well-conditioned enough
            break
        if attempt == max_attempts - 1:
            print(f"  Atentie: matricea generata are numar de conditie mare ({cond:.2e})")

    s = rng.uniform(low, high, size=n)
    return A, s

# ─────────────────────────────────────────────
# Helpers
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

def check(val, threshold=1e-6):
    return "✓ < 1e-6" if val < threshold else "✗ >= 1e-6"

def run_all_tasks(n, eps, A, s):
    """Run all 5 tasks given n, eps, A, s."""

    A_init = A.copy()

    # ── TASK 1 ──────────────────────────────
    print("\n" + "="*60)
    print("TASK 1: Calculul vectorului b = A * s")
    print("="*60)
    b = compute_b(A_init, s)
    b_init = b.copy()
    print_matrix("b", b, eps)

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
    print("\n" + "="*60)
    print("TASK 4: Erori de calcul")
    print("="*60)
    res_house, res_qr, rel_house, rel_qr = compute_errors(
        A_init, b_init, s, x_house, x_qr
    )
    print(f"\n  ||A * x_Householder - b||_2        = {res_house:.6e}  {check(res_house)}")
    print(f"  ||A * x_QR          - b||_2        = {res_qr:.6e}  {check(res_qr)}")
    print(f"\n  ||x_Householder - s||_2 / ||s||_2  = {rel_house:.6e}  {check(rel_house)}")
    print(f"  ||x_QR          - s||_2 / ||s||_2  = {rel_qr:.6e}  {check(rel_qr)}")

    # ── TASK 5 ──────────────────────────────
    print("\n" + "="*60)
    print("TASK 5: Inversa matricei A")
    print("="*60)
    A_inv_house = invert_via_householder_qr(A_init, Q_house, R_house, eps)
    A_inv_bibl  = np.linalg.inv(A_init)
    print_matrix("A^-1 (Householder)", A_inv_house, eps)
    print_matrix("A^-1 (numpy)",       A_inv_bibl,  eps)
    diff = np.linalg.norm(A_inv_house - A_inv_bibl)
    print(f"\n  ||A_inv_Householder - A_inv_bibl|| = {diff:.6e}  {check(diff)}")
    print(f"  ||A * A_inv_House - I||            = {np.linalg.norm(A_init @ A_inv_house - np.eye(n)):.6e}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════╗")
    print("║   QR Decomposition - All Tasks       ║")
    print("╚══════════════════════════════════════╝")

    n   = int(input("\nn = "))
    eps = float(input("eps = "))

    print("\nMod de initializare:")
    print("  [1] Introducere manuala")
    print("  [2] Generare aleatoare (random)")
    mode = input("Alegeti modul (1/2): ").strip()

    if mode == "2":
        # ── TASK 6: Random initialization ───
        seed_input = input("Seed (optional, Enter pentru random): ").strip()
        seed = int(seed_input) if seed_input else None

        range_input = input("Interval valori [min max] (Enter pentru [-10, 10]): ").strip()
        if range_input:
            low, high = map(float, range_input.split())
        else:
            low, high = -10.0, 10.0

        A, s = generate_random_data(n, seed=seed, value_range=(low, high))

        print(f"\n  Date generate aleator (seed={seed}, interval=[{low}, {high}]):")
        print_matrix("A (random)", A, eps)
        print_matrix("s (random)", s, eps)
        print(f"\n  Numarul de conditie al lui A: {np.linalg.cond(A):.4e}")

    else:
        # Manual input
        print(f"\nIntroduceti matricea A ({n}x{n}):")
        A = np.array([
            [float(input(f"  A[{i+1}][{j+1}] = ")) for j in range(n)]
            for i in range(n)
        ], dtype=float)

        print(f"\nIntroduceti vectorul s ({n} elemente):")
        s = np.array([float(input(f"  s[{j+1}] = ")) for j in range(n)])

    # Run all tasks
    run_all_tasks(n, eps, A, s)

if __name__ == "__main__":
    main()