"""Built-in test functions with known minima and analytical gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    """A named minimization problem."""

    name: str
    dim: int
    f: Callable[[Sequence[float]], float]
    grad: Callable[[Sequence[float]], list[float]]
    x_star: tuple[float, ...]
    f_star: float
    domain: tuple[tuple[float, ...], tuple[float, ...]]
    x0: tuple[float, ...]
    residuals: Callable[[Sequence[float]], list[float]] | None = None
    jacobian: Callable[[Sequence[float]], list[list[float]]] | None = None
    description: str = ""


def _quad1d(x: Sequence[float]) -> float:
    d = x[0] - 2.0
    return d * d


def _quad1d_g(x: Sequence[float]) -> list[float]:
    return [2.0 * (x[0] - 2.0)]


def _valley1d(x: Sequence[float]) -> float:
    """Narrow well: 100 (x-1)^2 — 1D analogue of an ill-conditioned bowl."""
    d = x[0] - 1.0
    return 100.0 * d * d


def _valley1d_g(x: Sequence[float]) -> list[float]:
    return [200.0 * (x[0] - 1.0)]


def _rosenbrock(x: Sequence[float]) -> float:
    a = 1.0 - x[0]
    b = x[1] - x[0] * x[0]
    return a * a + 100.0 * b * b


def _rosenbrock_g(x: Sequence[float]) -> list[float]:
    x0, x1 = x[0], x[1]
    dx = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0)
    dy = 200.0 * (x1 - x0 * x0)
    return [dx, dy]


def _rosenbrock_r(x: Sequence[float]) -> list[float]:
    return [1.0 - x[0], 10.0 * (x[1] - x[0] * x[0])]


def _rosenbrock_j(x: Sequence[float]) -> list[list[float]]:
    return [[-1.0, 0.0], [-20.0 * x[0], 10.0]]


def _stretched(x: Sequence[float]) -> float:
    return 100.0 * x[0] * x[0] + x[1] * x[1]


def _stretched_g(x: Sequence[float]) -> list[float]:
    return [200.0 * x[0], 2.0 * x[1]]


def _stretched_r(x: Sequence[float]) -> list[float]:
    return [10.0 * x[0], x[1]]


def _stretched_j(x: Sequence[float]) -> list[list[float]]:
    return [[10.0, 0.0], [0.0, 1.0]]


def _sphere(x: Sequence[float]) -> float:
    return x[0] * x[0] + x[1] * x[1]


def _sphere_g(x: Sequence[float]) -> list[float]:
    return [2.0 * x[0], 2.0 * x[1]]


def _sphere_r(x: Sequence[float]) -> list[float]:
    return [x[0], x[1]]


def _sphere_j(x: Sequence[float]) -> list[list[float]]:
    return [[1.0, 0.0], [0.0, 1.0]]


OBJECTIVES: dict[str, Objective] = {
    "quad1d": Objective(
        name="quad1d",
        dim=1,
        f=_quad1d,
        grad=_quad1d_g,
        x_star=(2.0,),
        f_star=0.0,
        domain=((-2.0,), (6.0,)),
        x0=(5.0,),
        description="(x-2)^2 — the textbook bowl",
    ),
    "valley1d": Objective(
        name="valley1d",
        dim=1,
        f=_valley1d,
        grad=_valley1d_g,
        x_star=(1.0,),
        f_star=0.0,
        domain=((-2.0,), (4.0,)),
        x0=(3.5,),
        description="100 (x-1)^2 — same min, much steeper walls",
    ),
    "rosenbrock": Objective(
        name="rosenbrock",
        dim=2,
        f=_rosenbrock,
        grad=_rosenbrock_g,
        x_star=(1.0, 1.0),
        f_star=0.0,
        domain=((-2.0, -1.0), (2.0, 3.0)),
        x0=(-1.2, 1.0),
        residuals=_rosenbrock_r,
        jacobian=_rosenbrock_j,
        description="banana valley (1-x)^2 + 100 (y-x^2)^2",
    ),
    "stretched": Objective(
        name="stretched",
        dim=2,
        f=_stretched,
        grad=_stretched_g,
        x_star=(0.0, 0.0),
        f_star=0.0,
        domain=((-2.0, -2.0), (2.0, 2.0)),
        x0=(1.5, 1.5),
        residuals=_stretched_r,
        jacobian=_stretched_j,
        description="100 x^2 + y^2 — skinny ellipse",
    ),
    "sphere": Objective(
        name="sphere",
        dim=2,
        f=_sphere,
        grad=_sphere_g,
        x_star=(0.0, 0.0),
        f_star=0.0,
        domain=((-2.0, -2.0), (2.0, 2.0)),
        x0=(1.5, -1.2),
        residuals=_sphere_r,
        jacobian=_sphere_j,
        description="x^2 + y^2 — isotropic bowl",
    ),
}


def get_objective(name: str) -> Objective:
    try:
        return OBJECTIVES[name]
    except KeyError as exc:
        known = ", ".join(sorted(OBJECTIVES))
        raise KeyError(f"unknown objective {name!r}; choose one of: {known}") from exc
