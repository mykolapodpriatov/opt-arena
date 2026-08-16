"""opt-arena: race classical minimization methods on one function."""

from __future__ import annotations

from opt_arena.functions import OBJECTIVES, Objective, get_objective
from opt_arena.report import render_csv, write_csv
from opt_arena.run import ALL_METHODS, METHODS_1D, METHODS_ND, race, run_method
from opt_arena.types import Budget, RunResult

__version__ = "0.1.0"

__all__ = [
    "ALL_METHODS",
    "METHODS_1D",
    "METHODS_ND",
    "OBJECTIVES",
    "Budget",
    "Objective",
    "RunResult",
    "__version__",
    "get_objective",
    "race",
    "render_csv",
    "run_method",
    "write_csv",
]
