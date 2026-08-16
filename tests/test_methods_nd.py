from __future__ import annotations

from opt_arena.functions import get_objective
from opt_arena.run import METHODS_ND, run_method
from opt_arena.types import Budget


def test_nd_methods_find_sphere() -> None:
    obj = get_objective("sphere")
    budget = Budget(max_iter=80, tol_g=1e-6, tol_x=1e-8)
    for name in METHODS_ND:
        result = run_method(name, obj, budget)
        assert result.path, name
        err = sum((a - b) ** 2 for a, b in zip(result.x_final, obj.x_star, strict=True)) ** 0.5
        assert err < 1e-3, (name, result.x_final, result.reason, result.n_f)
        assert result.f_final < 1e-6


def test_nd_methods_approach_rosenbrock() -> None:
    obj = get_objective("rosenbrock")
    budget = Budget(max_iter=250, max_f=8000, max_grad=8000, tol_g=1e-5)
    for name in METHODS_ND:
        result = run_method(name, obj, budget)
        err = sum((a - b) ** 2 for a, b in zip(result.x_final, obj.x_star, strict=True)) ** 0.5
        assert err < 0.15, (name, result.x_final, result.reason, result.n_f, result.f_final)
        assert result.f_final < 0.05, (name, result.f_final)


def test_stretched_quad_is_harder_than_sphere_for_cg() -> None:
    sphere = run_method("conjugate_grad", get_objective("sphere"), Budget(max_iter=80))
    skinny = run_method("conjugate_grad", get_objective("stretched"), Budget(max_iter=80))
    assert sphere.n_f > 0 and skinny.n_f > 0
    assert skinny.n_f >= sphere.n_f or skinny.f_final >= sphere.f_final
