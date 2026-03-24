import numpy as np

def compute_b(A, s):
    n = len(s)
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * A[i][j]
    return b

def main():
    n = int(input("n = "))
    eps = float(input("eps = "))

    print(f"Introduceti matricea A ({n}x{n}):")
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            val = float(input(f"A[{i+1}][{j+1}] = "))
            row.append(val)
        A.append(row)

    print(f"Introduceti vectorul s ({n} elemente):")
    s = []
    for j in range(n):
        val = float(input(f"s[{j+1}] = "))
        s.append(val)

    b = compute_b(A, s)

    print("\nVectorul b:")
    for i in range(n):
        if abs(b[i]) < eps:
            b[i] = 0.0
        print(f"b[{i+1}] = {b[i]:.6f}")

if __name__ == "__main__":
    main()