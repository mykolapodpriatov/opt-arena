"""Shared types for a single minimization run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Budget:
    """Hard caps and stopping tolerances for one method."""

    max_iter: int = 200
    max_f: int = 5000
    max_grad: int = 5000
    tol_x: float = 1e-8
    tol_f: float = 1e-12
    tol_g: float = 1e-8


@dataclass
class RunResult:
    """Trace of one method on one objective."""

    method: str
    path: list[list[float]] = field(default_factory=list)
    f_path: list[float] = field(default_factory=list)
    n_f: int = 0
    n_grad: int = 0
    x_final: list[float] = field(default_factory=list)
    f_final: float = float("inf")
    converged: bool = False
    reason: str = ""

    def record(self, x: list[float], f_val: float) -> None:
        self.path.append(list(x))
        self.f_path.append(f_val)
        self.x_final = list(x)
        self.f_final = f_val
