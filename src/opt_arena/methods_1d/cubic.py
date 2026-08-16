"""Cubic interpolation (Hermite / Davidon-style) using f and f'."""

from __future__ import annotations

import math

from opt_arena.counted import Counted
from opt_arena.types import Budget, RunResult


def _critical(a: float, b: float, fa: float, fb: float, ga: float, gb: float) -> float:
    if abs(b - a) < 1e-18:
        return 0.5 * (a + b)
    z = 3.0 * (fa - fb) / (b - a) + ga + gb
    disc = z * z - ga * gb
    if disc < 0.0:
        return 0.5 * (a + b)
    w = math.sqrt(disc)
    denom = gb - ga + 2.0 * w
    if abs(denom) < 1e-18:
        return 0.5 * (a + b)
    xmin = b - (b - a) * (gb + w - z) / denom
    lo, hi = (a, b) if a < b else (b, a)
    span = hi - lo
    return min(max(xmin, lo + 1e-12 * span), hi - 1e-12 * span)


def cubic(
    counted: Counted,
    lo: float,
    hi: float,
    budget: Budget,
) -> RunResult:
    result = RunResult(method="cubic")
    a, b = lo, hi
    fa = counted.f([a])
    fb = counted.f([b])
    ga = counted.grad([a])[0]
    gb = counted.grad([b])[0]
    result.record([a], fa)

    for _ in range(budget.max_iter):
        if counted.exhausted(budget) or abs(b - a) <= budget.tol_x:
            x = a if fa < fb else b
            result.record([x], fa if fa < fb else fb)
            result.converged = abs(b - a) <= budget.tol_x
            result.reason = "tol_x" if result.converged else "budget"
            return result
        xmin = _critical(a, b, fa, fb, ga, gb)
        fx = counted.f([xmin])
        gx = counted.grad([xmin])[0]
        result.record([xmin], fx)
        if abs(gx) <= budget.tol_g:
            result.converged = True
            result.reason = "tol_g"
            return result
        if gx > 0.0:
            b, fb, gb = xmin, fx, gx
        else:
            a, fa, ga = xmin, fx, gx

    result.reason = "max_iter"
    return result
