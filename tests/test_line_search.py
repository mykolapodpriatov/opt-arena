"""Armijo backtracking rejects a step it never accepted."""

from __future__ import annotations

from opt_arena.counted import Counted
from opt_arena.line_search import armijo_backtrack
from opt_arena.types import Budget


def test_exhausted_backtrack_does_not_return_an_untested_step() -> None:
    # f(x) = -x + 1e12 x^2. From 0, direction +1, thirty halvings from 1
    # still sit above the Armijo bound. The old return value was the next
    # untested shrink, and f increased along it.
    def f(x: list[float]) -> float:
        v = x[0]
        return -v + 1e12 * v * v

    counted = Counted(f)
    alpha = armijo_backtrack(
        counted,
        [0.0],
        0.0,
        [-1.0],
        [1.0],
        Budget(),
        alpha0=1.0,
        c=1e-4,
        shrink=0.5,
        max_steps=30,
    )
    assert alpha == 0.0
