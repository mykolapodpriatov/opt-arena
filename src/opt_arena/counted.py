"""Wrap f / grad so every evaluation is counted against a budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from opt_arena.types import Budget

Vec = list[float]
Fun = Callable[[Sequence[float]], float]
Grad = Callable[[Sequence[float]], list[float]]


class Counted:
    """Counts calls; `exhausted` is true once any budget cap is hit."""

    def __init__(self, f: Fun, grad: Grad | None = None) -> None:
        self._f = f
        self._grad = grad
        self.n_f = 0
        self.n_grad = 0

    def f(self, x: Sequence[float]) -> float:
        self.n_f += 1
        return self._f(x)

    def grad(self, x: Sequence[float]) -> list[float]:
        if self._grad is None:
            raise RuntimeError("this objective has no analytical gradient")
        self.n_grad += 1
        return self._grad(x)

    def exhausted(self, budget: Budget) -> bool:
        return self.n_f >= budget.max_f or self.n_grad >= budget.max_grad
