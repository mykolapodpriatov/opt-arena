"""Levenberg-Marquardt for least-squares objectives (residuals + Jacobian)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from opt_arena.counted import Counted
from opt_arena.linalg import add, dot, matmul, matvec, norm, scale, solve_spd, transpose
from opt_arena.types import Budget, RunResult

Residual = Callable[[Sequence[float]], list[float]]
Jacobian = Callable[[Sequence[float]], list[list[float]]]


def levenberg(
    counted: Counted,
    x0: Sequence[float],
    budget: Budget,
    residuals: Residual,
    jacobian: Jacobian,
    *,
    mu0: float = 1e-3,
) -> RunResult:
    result = RunResult(method="levenberg")
    x = list(x0)
    n = len(x)
    mu = mu0
    r = residuals(x)
    counted.n_f += 1
    fval = dot(r, r)
    result.record(x, fval)

    for _ in range(budget.max_iter):
        if counted.exhausted(budget):
            result.reason = "budget"
            return result
        jac = jacobian(x)
        counted.n_grad += 1
        jt = transpose(jac)
        jtj = matmul(jt, jac)
        g = matvec(jt, r)
        if norm(g) <= budget.tol_g:
            result.converged = True
            result.reason = "tol_g"
            return result
        accepted = False
        for _inner in range(8):
            damped = [list(row) for row in jtj]
            for i in range(n):
                damped[i][i] += mu
            try:
                step = scale(solve_spd(damped, g), -1.0)
            except ZeroDivisionError:
                mu *= 10.0
                continue
            x_try = add(x, step)
            r_try = residuals(x_try)
            counted.n_f += 1
            f_try = dot(r_try, r_try)
            if f_try < fval:
                x, r, fval = x_try, r_try, f_try
                result.record(x, fval)
                mu = max(mu / 3.0, 1e-12)
                accepted = True
                if norm(step) <= budget.tol_x:
                    result.converged = True
                    result.reason = "tol_x"
                    return result
                break
            mu *= 10.0
            if counted.exhausted(budget):
                result.reason = "budget"
                return result
        if not accepted:
            result.reason = "no_descent"
            return result

    result.reason = "max_iter"
    return result
