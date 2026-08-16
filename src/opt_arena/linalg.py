"""Tiny vector helpers — no NumPy in the core."""

from __future__ import annotations

from collections.abc import Sequence


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def scale(a: Sequence[float], s: float) -> list[float]:
    return [s * x for x in a]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


def norm2(a: Sequence[float]) -> float:
    return dot(a, a)


def norm(a: Sequence[float]) -> float:
    value: float = norm2(a) ** 0.5
    return value


def identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matvec(m: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [dot(row, v) for row in m]


def transpose(m: Sequence[Sequence[float]]) -> list[list[float]]:
    n, k = len(m), len(m[0])
    return [[m[i][j] for i in range(n)] for j in range(k)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def solve_spd(a_in: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small n only."""
    n = len(b)
    a = [[*list(row), b[i]] for i, row in enumerate(a_in)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        diag = a[col][col]
        if abs(diag) < 1e-18:
            raise ZeroDivisionError("singular linear system")
        for r in range(col + 1, n):
            factor = a[r][col] / diag
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = a[i][n]
        for j in range(i + 1, n):
            acc -= a[i][j] * x[j]
        x[i] = acc / a[i][i]
    return x
