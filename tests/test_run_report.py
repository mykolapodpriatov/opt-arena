from __future__ import annotations

import json
from pathlib import Path

import pytest

from opt_arena.cli import main
from opt_arena.report import rank_results, render_csv, render_verdict
from opt_arena.run import race
from opt_arena.types import Budget, RunResult


def test_race_auto_picks_1d_methods() -> None:
    results = race("quad1d")
    names = [r.method for r in results]
    assert names == ["dichotomy", "golden", "cubic"]


def test_race_auto_picks_nd_methods() -> None:
    results = race("sphere")
    names = [r.method for r in results]
    assert names == ["conjugate_grad", "newton_quasi", "levenberg"]


def test_csv_has_header_and_rows() -> None:
    results = race("quad1d")
    text = render_csv(results, objective="quad1d")
    lines = [ln for ln in text.strip().splitlines() if ln]
    assert lines[0].startswith("rank,method,")
    assert len(lines) == 4


def test_cli_json(capsys: object) -> None:
    code = main(["run", "--function", "sphere", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["function"] == "sphere"
    assert len(payload["results"]) == 3


def test_cli_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "race.csv"
    code = main(["run", "--function", "quad1d", "--csv", str(out)])
    assert code == 0
    assert out.is_file()
    assert "golden" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# ranking and verdict
# --------------------------------------------------------------------------- #


def _result(
    method: str,
    *,
    converged: bool,
    n_f: int = 0,
    f_final: float = 0.0,
    x_final: list[float] | None = None,
) -> RunResult:
    return RunResult(
        method=method,
        converged=converged,
        n_f=n_f,
        f_final=f_final,
        x_final=x_final if x_final is not None else [0.0, 0.0],
        path=[x_final if x_final is not None else [0.0, 0.0]],
    )


def test_converged_beats_unconverged_whatever_the_final_f() -> None:
    """A method that ran out of budget at a lower f did not win: it did not
    stop, it was stopped."""
    lucky_loser = _result("lucky", converged=False, n_f=1, f_final=0.0)
    real_winner = _result("real", converged=True, n_f=999, f_final=1.0)

    order = [r.method for r in rank_results([lucky_loser, real_winner])]

    assert order == ["real", "lucky"]


def test_fewer_function_evaluations_wins_among_converged() -> None:
    cheap = _result("cheap", converged=True, n_f=10, f_final=1e-9)
    dear = _result("dear", converged=True, n_f=400, f_final=1e-12)

    order = [r.method for r in rank_results([dear, cheap])]

    assert order == ["cheap", "dear"]


def test_distance_to_the_optimum_breaks_an_n_f_tie() -> None:
    near = _result("near", converged=True, n_f=10, x_final=[0.0, 0.0])
    far = _result("far", converged=True, n_f=10, x_final=[5.0, 5.0])

    order = [r.method for r in rank_results([far, near], objective="sphere")]

    assert order == ["near", "far"]


def test_method_name_breaks_the_last_tie_so_runs_agree() -> None:
    b = _result("b_method", converged=True, n_f=10)
    a = _result("a_method", converged=True, n_f=10)

    order = [r.method for r in rank_results([b, a], objective="sphere")]

    assert order == ["a_method", "b_method"]


def test_unconverged_methods_are_ordered_by_final_f() -> None:
    """None of them arrived, so all that is left is who got lowest."""
    high = _result("high", converged=False, f_final=10.0)
    low = _result("low", converged=False, f_final=0.5)

    order = [r.method for r in rank_results([high, low])]

    assert order == ["low", "high"]


def test_a_method_that_produced_no_point_still_ranks() -> None:
    """A method rejected as inapplicable must appear in the table, not vanish."""
    nothing = RunResult(method="levenberg", reason="needs residuals")
    ran = _result("golden", converged=True, n_f=12, x_final=[2.0])

    order = [r.method for r in rank_results([nothing, ran], objective="quad1d")]

    assert order == ["golden", "levenberg"]


def test_ranking_is_stable_across_runs() -> None:
    results = race("sphere")

    first = [r.method for r in rank_results(results, objective="sphere")]
    second = [r.method for r in rank_results(results, objective="sphere")]

    assert first == second


def test_verdict_names_the_winner() -> None:
    results = race("quad1d")

    line = render_verdict(results, objective="quad1d")

    assert line.startswith("Winner: ")
    assert "function evaluation" in line


def test_verdict_says_so_when_nobody_converged() -> None:
    nobody = [
        _result("a", converged=False, f_final=3.0),
        _result("b", converged=False, f_final=1.0),
    ]

    line = render_verdict(nobody)

    assert line.startswith("Nobody converged.")
    assert "b" in line


def test_verdict_handles_an_empty_race() -> None:
    assert render_verdict([]) == "No methods ran.\n"


# --------------------------------------------------------------------------- #
# budget flags
# --------------------------------------------------------------------------- #


def test_every_budget_flag_reaches_the_run(monkeypatch: object) -> None:
    seen: dict[str, Budget] = {}

    def fake_race(name, methods=None, budget=None, x0=None):  # type: ignore[no-untyped-def]
        seen["budget"] = budget
        return []

    monkeypatch.setattr("opt_arena.cli.race", fake_race)  # type: ignore[attr-defined]
    code = main(
        [
            "run",
            "--function",
            "sphere",
            "--max-iter",
            "11",
            "--max-f",
            "22",
            "--max-grad",
            "33",
            "--tol-x",
            "1e-3",
            "--tol-f",
            "1e-4",
            "--tol-g",
            "1e-5",
        ]
    )

    assert code == 0
    budget = seen["budget"]
    assert budget.max_iter == 11
    assert budget.max_f == 22
    assert budget.max_grad == 33
    assert budget.tol_x == 1e-3
    assert budget.tol_f == 1e-4
    assert budget.tol_g == 1e-5


def test_budget_defaults_match_the_dataclass(monkeypatch: object) -> None:
    """Restating the defaults in add_argument would let them drift."""
    seen: dict[str, Budget] = {}

    def fake_race(name, methods=None, budget=None, x0=None):  # type: ignore[no-untyped-def]
        seen["budget"] = budget
        return []

    monkeypatch.setattr("opt_arena.cli.race", fake_race)  # type: ignore[attr-defined]
    main(["run", "--function", "sphere"])

    assert seen["budget"] == Budget()


def test_a_tight_function_budget_actually_stops_the_run() -> None:
    """The flag has to bound the race, not just be accepted."""
    generous = race("rosenbrock", budget=Budget(max_f=5000))
    tight = race("rosenbrock", budget=Budget(max_f=12))

    assert max(r.n_f for r in tight) <= max(r.n_f for r in generous)
    assert any(not r.converged for r in tight)


@pytest.mark.parametrize(
    "args",
    [
        ["--max-iter", "0"],
        ["--max-f", "0"],
        ["--max-grad", "-1"],
        ["--tol-x", "0"],
        ["--tol-f", "-1e-9"],
        ["--tol-g", "0"],
    ],
)
def test_a_useless_budget_is_a_usage_error(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--function", "sphere", *args])
    assert exc.value.code == 2


def test_json_carries_the_rank(capsys: object) -> None:
    main(["run", "--function", "sphere", "--json"])
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    ranks = [r["rank"] for r in payload["results"]]
    assert ranks == list(range(1, len(ranks) + 1))
