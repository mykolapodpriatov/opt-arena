"""Golden-section search on a unimodal interval."""

from __future__ import annotations

from opt_arena.counted import Counted
from opt_arena.types import Budget, RunResult

_PHI = 0.5 * (5.0**0.5 - 1.0)


def golden(
    counted: Counted,
    lo: float,
    hi: float,
    budget: Budget,
) -> RunResult:
    result = RunResult(method="golden")
    a, b = lo, hi
    c = b - _PHI * (b - a)
    d = a + _PHI * (b - a)
    fc = counted.f([c])
    fd = counted.f([d])
    result.record([c], fc)

    for _ in range(budget.max_iter):
        if counted.exhausted(budget) or (b - a) <= budget.tol_x:
            x = c if fc < fd else d
            fx = fc if fc < fd else fd
            result.record([x], fx)
            result.converged = (b - a) <= budget.tol_x
            result.reason = "tol_x" if result.converged else "budget"
            return result
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _PHI * (b - a)
            fc = counted.f([c])
            result.record([c], fc)
        else:
            a, c, fc = c, d, fd
            d = a + _PHI * (b - a)
            fd = counted.f([d])
            result.record([d], fd)

    result.reason = "max_iter"
    return result
