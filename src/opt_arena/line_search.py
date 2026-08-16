"""Armijo backtracking used by the nD methods."""

from __future__ import annotations

from collections.abc import Sequence

from opt_arena.counted import Counted
from opt_arena.linalg import add, dot, scale
from opt_arena.types import Budget


def armijo_backtrack(
    counted: Counted,
    x: Sequence[float],
    f0: float,
    g: Sequence[float],
    direction: Sequence[float],
    budget: Budget,
    *,
    alpha0: float = 1.0,
    c: float = 1e-4,
    shrink: float = 0.5,
    max_steps: int = 30,
) -> float:
    slope = dot(g, direction)
    if slope >= 0.0:
        return 0.0
    alpha = alpha0
    for _ in range(max_steps):
        if counted.exhausted(budget):
            return alpha
        trial = add(x, scale(direction, alpha))
        fi = counted.f(trial)
        if fi <= f0 + c * alpha * slope:
            return alpha
        alpha *= shrink
    return alpha
