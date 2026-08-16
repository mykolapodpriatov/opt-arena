"""Dichotomy (two-point interval reduction) for a unimodal 1D function."""

from __future__ import annotations

from opt_arena.counted import Counted
from opt_arena.types import Budget, RunResult


def dichotomy(
    counted: Counted,
    lo: float,
    hi: float,
    budget: Budget,
) -> RunResult:
    result = RunResult(method="dichotomy")
    a, b = lo, hi
    mid = 0.5 * (a + b)
    result.record([mid], counted.f([mid]))

    for _ in range(budget.max_iter):
        if counted.exhausted(budget) or (b - a) <= budget.tol_x:
            result.converged = (b - a) <= budget.tol_x
            result.reason = "tol_x" if result.converged else "budget"
            return result
        probe = max((b - a) * 0.25, budget.tol_x * 0.5)
        x1 = 0.5 * (a + b) - probe
        x2 = 0.5 * (a + b) + probe
        f1 = counted.f([x1])
        if counted.exhausted(budget):
            result.reason = "budget"
            return result
        f2 = counted.f([x2])
        if f1 < f2:
            b = x2
            result.record([x1], f1)
        else:
            a = x1
            result.record([x2], f2)

    result.reason = "max_iter"
    return result
