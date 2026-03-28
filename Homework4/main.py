from pathlib import Path
from typing import Iterable


def read_dimension(lines: Iterable[str]) -> int:
    """Return count of non-empty lines parsed as floats."""
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        float(stripped)  # Validate parse; value not needed beyond count.
        count += 1
    return count


def read_floats(lines: Iterable[str]) -> list[float]:
    """Parse non-empty lines into floats."""
    values: list[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # --- Exercise 1 ---
    print("--- Exercise 1 ---")
    for i in range(1, 6):
        file_path = base_dir / f"d0_{i}.txt"
        with file_path.open("r", encoding="utf-8") as f:
            dimension = read_dimension(f)
        print(f"System {i}: dimension = {dimension}")

    # --- Exercise 2 ---
    print("--- Exercise 2 ---")
    for i in range(1, 6):
        d0_path = base_dir / f"d0_{i}.txt"
        d1_path = base_dir / f"d1_{i}.txt"
        d2_path = base_dir / f"d2_{i}.txt"

        with (
            d0_path.open("r", encoding="utf-8") as d0_file,
            d1_path.open("r", encoding="utf-8") as d1_file,
            d2_path.open("r", encoding="utf-8") as d2_file,
        ):
            n = read_dimension(d0_file)
            p = n - read_dimension(d1_file)
            q = n - read_dimension(d2_file)

        print(f"System {i}: p = {p}, q = {q}")

    # --- Exercise 3 ---
    print("--- Exercise 3 ---")
    epsilon = 1e-10
    for i in range(1, 6):
        d0_path = base_dir / f"d0_{i}.txt"
        with d0_path.open("r", encoding="utf-8") as d0_file:
            d0_values = read_floats(d0_file)

        has_zero = any(abs(val) <= epsilon for val in d0_values)
        if has_zero:
            print("System {0}: INVALID (zero element on main diagonal)".format(i))
        else:
            print(f"System {i}: OK")

    # --- Exercise 4 ---
    print("--- Exercise 4 ---")
    epsilon = 1e-10
    divergence_threshold = 1e10
    kmax = 10000

    x_solutions: list[list[float] | None] = [None] * 5
    b_vectors: list[list[float] | None] = [None] * 5
    y_vectors: list[list[float] | None] = [None] * 5
    for i in range(1, 6):
        d0_path = base_dir / f"d0_{i}.txt"
        d1_path = base_dir / f"d1_{i}.txt"
        d2_path = base_dir / f"d2_{i}.txt"
        b_path = base_dir / f"b_{i}.txt"

        with (
            d0_path.open("r", encoding="utf-8") as d0_file,
            d1_path.open("r", encoding="utf-8") as d1_file,
            d2_path.open("r", encoding="utf-8") as d2_file,
            b_path.open("r", encoding="utf-8") as b_file,
        ):
            d0_vals = read_floats(d0_file)
            d1_vals = read_floats(d1_file)
            d2_vals = read_floats(d2_file)
            b_vals = read_floats(b_file)

        b_vectors[i - 1] = b_vals

        n = len(d0_vals)
        p = n - len(d1_vals)
        q = n - len(d2_vals)

        x = [0.0] * n
        converged = False

        for k in range(1, kmax + 1):
            max_diff = 0.0
            for idx in range(n):
                value = b_vals[idx]

                if idx - p >= 0:
                    value -= d1_vals[idx - p] * x[idx - p]
                if idx + p < n:
                    value -= d1_vals[idx] * x[idx + p]

                if idx - q >= 0:
                    value -= d2_vals[idx - q] * x[idx - q]
                if idx + q < n:
                    value -= d2_vals[idx] * x[idx + q]

                new_x = value / d0_vals[idx]
                diff = abs(new_x - x[idx])
                if diff > max_diff:
                    max_diff = diff
                x[idx] = new_x

            if max_diff < epsilon:
                converged = True
                print(f"System {i}: converged in {k} iterations")
                x_solutions[i - 1] = x.copy()
                break

            if max_diff > divergence_threshold:
                print(f"System {i}: divergence")
                break

        if not converged and max_diff <= divergence_threshold:
            print(f"System {i}: divergence")

    # --- Exercise 5 ---
    print("--- Exercise 5 ---")
    for i in range(1, 6):
        x_gs = x_solutions[i - 1]
        if x_gs is None:
            print(f"System {i}: skipped (no approximated solution)")
            continue

        d0_path = base_dir / f"d0_{i}.txt"
        d1_path = base_dir / f"d1_{i}.txt"
        d2_path = base_dir / f"d2_{i}.txt"

        with (
            d0_path.open("r", encoding="utf-8") as d0_file,
            d1_path.open("r", encoding="utf-8") as d1_file,
            d2_path.open("r", encoding="utf-8") as d2_file,
        ):
            d0_vals = read_floats(d0_file)
            d1_vals = read_floats(d1_file)
            d2_vals = read_floats(d2_file)

        n = len(d0_vals)
        p = n - len(d1_vals)
        q = n - len(d2_vals)

        y: list[float] = []
        for j in range(n):
            total = d0_vals[j] * x_gs[j]
            if j - p >= 0:
                total += d1_vals[j - p] * x_gs[j - p]
            if j + p < n:
                total += d1_vals[j] * x_gs[j + p]

            if j - q >= 0:
                total += d2_vals[j - q] * x_gs[j - q]
            if j + q < n:
                total += d2_vals[j] * x_gs[j + q]

            y.append(total)

        y_vectors[i - 1] = y
        print(f"System {i}: y = {y}")

    # --- Exercise 7 ---
    print("--- Exercise 7 ---")
    for i in range(1, 6):
        x_gs = x_solutions[i - 1]
        y_vals = y_vectors[i - 1]
        b_vals = b_vectors[i - 1]

        if x_gs is None or y_vals is None or b_vals is None:
            print(f"System {i}: skipped (no convergence)")
            continue

        max_diff = 0.0
        for y_val, b_val in zip(y_vals, b_vals):
            diff = abs(y_val - b_val)
            if diff > max_diff:
                max_diff = diff

        print(f"System {i}: ||AxGS - b||_inf = {max_diff}")


if __name__ == "__main__":
    main()
