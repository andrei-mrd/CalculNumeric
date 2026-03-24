import numpy as np

def householder_qr(A, eps):
    """
    QR decomposition using Householder reflections.
    Returns Q (orthogonal) and R (upper triangular).
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for k in range(n):
        # Extract the column vector below (and including) diagonal
        x = R[k:, k]

        # Compute the Householder vector v
        norm_x = np.linalg.norm(x)
        if norm_x < eps:
            continue  # skip if column already zeroed

        # Sign chosen to avoid cancellation
        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x

        # Normalize v
        v = v / np.linalg.norm(v)

        # Apply Householder reflection to R: R = H * R
        # H = I - 2*v*v^T, but we apply it efficiently
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])

        # Accumulate Q: Q = Q * H^T (H is symmetric so H^T = H)
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    # Apply epsilon threshold
    R[np.abs(R) < eps] = 0.0

    return Q, R

def print_matrix(name, M, eps):
    m, n = M.shape
    print(f"\nMatricea {name}:")
    for i in range(m):
        row = ""
        for j in range(n):
            val = 0.0 if abs(M[i][j]) < eps else M[i][j]
            row += f"{val:10.4f} "
        print(row)

def main():
    n = int(input("n = "))
    eps = float(input("eps = "))

    print(f"\nIntroduceti matricea A ({n}x{n}):")
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            val = float(input(f"A[{i+1}][{j+1}] = "))
            row.append(val)
        A.append(row)

    A = np.array(A, dtype=float)

    print("\nMatricea A introdusa:")
    print_matrix("A", A, eps)

    Q, R = householder_qr(A, eps)

    print_matrix("Q", Q, eps)
    print_matrix("R", R, eps)

    # Verify: A should equal Q * R
    A_reconstructed = Q @ R
    error = np.linalg.norm(A - A_reconstructed)
    print(f"\nEroarea de reconstructie ||A - Q*R|| = {error:.2e}")

    # Verify Q is orthogonal: Q^T * Q should be identity
    QtQ = Q.T @ Q
    orth_error = np.linalg.norm(QtQ - np.eye(n))
    print(f"Eroarea de ortogonalitate ||Q^T*Q - I|| = {orth_error:.2e}")

if __name__ == "__main__":
    main()