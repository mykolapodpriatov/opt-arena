"""Run one or every method on a named objective."""

from __future__ import annotations

from collections.abc import Sequence

from opt_arena.counted import Counted
from opt_arena.functions import Objective, get_objective
from opt_arena.methods_1d import cubic, dichotomy, golden
from opt_arena.methods_nd import conjugate_grad, levenberg, newton_quasi
from opt_arena.types import Budget, RunResult

METHODS_1D = ("dichotomy", "golden", "cubic")
METHODS_ND = ("conjugate_grad", "newton_quasi", "levenberg")
ALL_METHODS = METHODS_1D + METHODS_ND


def run_method(
    method: str,
    objective: Objective,
    budget: Budget | None = None,
    x0: Sequence[float] | None = None,
) -> RunResult:
    budget = budget or Budget()
    counted = Counted(objective.f, objective.grad)

    if method in METHODS_1D:
        if objective.dim != 1:
            return RunResult(
                method=method,
                reason=f"{method} is 1D; objective {objective.name} has dim {objective.dim}",
            )
        lo, hi = objective.domain[0][0], objective.domain[1][0]
        if method == "dichotomy":
            result = dichotomy(counted, lo, hi, budget)
        elif method == "golden":
            result = golden(counted, lo, hi, budget)
        else:
            result = cubic(counted, lo, hi, budget)
    elif method in METHODS_ND:
        if objective.dim < 2 and method == "levenberg":
            return RunResult(method=method, reason="levenberg needs residuals in dim>=2")
        start = list(x0) if x0 is not None else list(objective.x0)
        if method == "conjugate_grad":
            result = conjugate_grad(counted, start, budget)
        elif method == "newton_quasi":
            result = newton_quasi(counted, start, budget)
        else:
            if objective.residuals is None or objective.jacobian is None:
                return RunResult(
                    method=method,
                    reason=f"{objective.name} has no residual/jacobian for LM",
                )
            result = levenberg(
                counted,
                start,
                budget,
                objective.residuals,
                objective.jacobian,
            )
    else:
        raise KeyError(f"unknown method {method!r}; choose one of: {', '.join(ALL_METHODS)}")

    result.n_f = counted.n_f
    result.n_grad = counted.n_grad
    return result


def race(
    name: str,
    methods: Sequence[str] | None = None,
    budget: Budget | None = None,
    x0: Sequence[float] | None = None,
) -> list[RunResult]:
    objective = get_objective(name)
    if methods is None:
        chosen = list(METHODS_1D if objective.dim == 1 else METHODS_ND)
    else:
        chosen = list(methods)
    return [run_method(m, objective, budget, x0) for m in chosen]
