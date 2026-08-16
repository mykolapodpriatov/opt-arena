from __future__ import annotations

from opt_arena.functions import get_objective
from opt_arena.run import METHODS_1D, run_method
from opt_arena.types import Budget


def test_1d_methods_find_quad_minimum() -> None:
    obj = get_objective("quad1d")
    budget = Budget(max_iter=80, tol_x=1e-6)
    for name in METHODS_1D:
        result = run_method(name, obj, budget)
        assert result.path, name
        assert abs(result.x_final[0] - obj.x_star[0]) < 5e-3, (name, result)
        assert result.n_f > 0


def test_1d_methods_find_valley() -> None:
    obj = get_objective("valley1d")
    budget = Budget(max_iter=80, tol_x=1e-6)
    for name in METHODS_1D:
        result = run_method(name, obj, budget)
        assert abs(result.x_final[0] - 1.0) < 5e-3, (name, result)


def test_1d_rejects_2d_objective() -> None:
    obj = get_objective("rosenbrock")
    result = run_method("golden", obj)
    assert not result.converged
    assert "1D" in result.reason
