from __future__ import annotations

from opt_arena.counted import Counted
from opt_arena.functions import get_objective
from opt_arena.methods_1d.dichotomy import dichotomy
from opt_arena.methods_1d.golden import golden
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


def test_inverted_bracket_searches_the_same_interval() -> None:
    obj = get_objective("quad1d")
    budget = Budget(max_iter=80, tol_x=1e-6)
    lo, hi = obj.domain[0][0], obj.domain[1][0]
    for name, method in (("dichotomy", dichotomy), ("golden", golden)):
        ordered_count = Counted(obj.f, obj.grad)
        flipped_count = Counted(obj.f, obj.grad)
        ordered = method(ordered_count, lo, hi, budget)
        flipped = method(flipped_count, hi, lo, budget)
        assert abs(flipped.x_final[0] - obj.x_star[0]) < 5e-3, name
        assert flipped_count.n_f > 2, name
        assert flipped_count.n_f == ordered_count.n_f, name
        assert flipped.reason == ordered.reason, name
        assert abs(flipped.x_final[0] - ordered.x_final[0]) < 1e-9, name


def test_1d_rejects_2d_objective() -> None:
    obj = get_objective("rosenbrock")
    result = run_method("golden", obj)
    assert not result.converged
    assert "1D" in result.reason
