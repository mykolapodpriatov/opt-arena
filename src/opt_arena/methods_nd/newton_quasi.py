"""BFGS quasi-Newton with Armijo line search."""

from __future__ import annotations

from collections.abc import Sequence

from opt_arena.counted import Counted
from opt_arena.linalg import add, dot, identity, matvec, norm, scale, sub
from opt_arena.line_search import armijo_backtrack
from opt_arena.types import Budget, RunResult


def _bfgs_update(
    h: list[list[float]],
    s: Sequence[float],
    y: Sequence[float],
) -> list[list[float]]:
    n = len(s)
    ys = dot(y, s)
    if abs(ys) < 1e-16:
        return h
    rho = 1.0 / ys
    i_rho_sy = [
        [((1.0 if i == j else 0.0) - rho * s[i] * y[j]) for j in range(n)] for i in range(n)
    ]
    i_rho_ys = [
        [((1.0 if i == j else 0.0) - rho * y[i] * s[j]) for j in range(n)] for i in range(n)
    ]
    tmp = [[sum(i_rho_sy[i][k] * h[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    updated = [
        [sum(tmp[i][k] * i_rho_ys[k][j] for k in range(n)) + rho * s[i] * s[j] for j in range(n)]
        for i in range(n)
    ]
    return updated


def newton_quasi(
    counted: Counted,
    x0: Sequence[float],
    budget: Budget,
) -> RunResult:
    result = RunResult(method="newton_quasi")
    x = list(x0)
    n = len(x)
    h = identity(n)
    f0 = counted.f(x)
    g = counted.grad(x)
    result.record(x, f0)

    for _ in range(budget.max_iter):
        if counted.exhausted(budget):
            result.reason = "budget"
            return result
        if norm(g) <= budget.tol_g:
            result.converged = True
            result.reason = "tol_g"
            return result
        direction = scale(matvec(h, g), -1.0)
        if dot(direction, g) >= 0.0:
            direction = scale(g, -1.0)
            h = identity(n)
        alpha = armijo_backtrack(counted, x, result.f_final, g, direction, budget)
        if alpha <= 0.0:
            result.reason = "no_descent"
            return result
        s = scale(direction, alpha)
        x_new = add(x, s)
        f_new = counted.f(x_new)
        g_new = counted.grad(x_new)
        result.record(x_new, f_new)
        if norm(s) <= budget.tol_x:
            result.converged = True
            result.reason = "tol_x"
            return result
        y = sub(g_new, g)
        h = _bfgs_update(h, s, y)
        x, g = x_new, g_new

    result.reason = "max_iter"
    return result
