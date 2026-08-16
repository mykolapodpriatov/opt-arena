"""Fletcher-Reeves conjugate gradient with Armijo line search."""

from __future__ import annotations

from collections.abc import Sequence

from opt_arena.counted import Counted
from opt_arena.linalg import add, dot, norm, scale
from opt_arena.line_search import armijo_backtrack
from opt_arena.types import Budget, RunResult


def conjugate_grad(
    counted: Counted,
    x0: Sequence[float],
    budget: Budget,
) -> RunResult:
    result = RunResult(method="conjugate_grad")
    x = list(x0)
    f0 = counted.f(x)
    g = counted.grad(x)
    d = scale(g, -1.0)
    result.record(x, f0)

    for _ in range(budget.max_iter):
        if counted.exhausted(budget):
            result.reason = "budget"
            return result
        if norm(g) <= budget.tol_g:
            result.converged = True
            result.reason = "tol_g"
            return result
        alpha = armijo_backtrack(counted, x, result.f_final, g, d, budget)
        if alpha <= 0.0:
            result.reason = "no_descent"
            return result
        x_new = add(x, scale(d, alpha))
        step = norm(scale(d, alpha))
        f_new = counted.f(x_new)
        g_new = counted.grad(x_new)
        result.record(x_new, f_new)
        if step <= budget.tol_x or abs(f_new - result.f_path[-2]) <= budget.tol_f:
            result.converged = True
            result.reason = "tol_x" if step <= budget.tol_x else "tol_f"
            return result
        denom = dot(g, g)
        beta = 0.0 if denom <= 1e-18 else dot(g_new, g_new) / denom
        d = add(scale(g_new, -1.0), scale(d, beta))
        if dot(d, g_new) >= 0.0:
            d = scale(g_new, -1.0)
        x, g = x_new, g_new

    result.reason = "max_iter"
    return result
