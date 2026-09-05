"""CSV comparison of a race, plus the ranking that decides who won."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from opt_arena.functions import get_objective
from opt_arena.types import RunResult

#: Sentinel distance for a method that never produced a point, so it sorts last
#: among its own group rather than being dropped from the ranking.
_NO_POINT = float("inf")


def error_to_optimum(result: RunResult, objective: str) -> float:
    """Euclidean distance from the final point to the objective's known ``x*``.

    ``inf`` when the method never produced a point, when the objective is
    unknown, or when the two disagree on dimension: a method that got nowhere
    must not tie-break its way up the table, and ranking must never raise.
    """
    if not result.x_final or not objective:
        return _NO_POINT
    obj = get_objective(objective)
    if len(result.x_final) != len(obj.x_star):
        return _NO_POINT
    total: float = 0.0
    for a, b in zip(result.x_final, obj.x_star, strict=True):
        total += (a - b) ** 2
    return math.sqrt(total)


def rank_results(results: Sequence[RunResult], *, objective: str = "") -> list[RunResult]:
    """Order a race best-first.

    The ordering is the whole point of the tool, so it is spelled out rather
    than left to whatever a sort key happens to do:

    1. **Converged before unconverged.** A method that stopped on a tolerance
       beat one that ran out of budget, whatever their final ``f``. Comparing
       those two by ``f_final`` would reward a method for being lucky about
       where its budget ran out.
    2. **Among converged, fewer function evaluations first.** Iterations are not
       comparable across method classes (a CG step costs a whole line search;
       a dichotomy step costs two evaluations), which is why ``n_f`` is the
       cost axis this project already reports.
    3. **Then distance to the known optimum, then the method name**, so two runs
       of the same race agree.

    Among unconverged methods none of them arrived, so all that is left is who
    got lowest: they are ordered by ``f_final``, then by name.
    """

    def key(result: RunResult) -> tuple[object, ...]:
        if result.converged:
            return (0, result.n_f, error_to_optimum(result, objective), result.method)
        return (1, result.f_final, error_to_optimum(result, objective), result.method)

    return sorted(results, key=key)


def render_verdict(results: Sequence[RunResult], *, objective: str = "") -> str:
    """One line naming the winner and the reason it won, or why nobody did."""
    ranked = rank_results(results, objective=objective)
    if not ranked:
        return "No methods ran.\n"
    best = ranked[0]
    if not best.converged:
        return (
            f"Nobody converged. Closest: {best.method} at f={best.f_final:.6g} "
            f"({best.reason or 'no reason recorded'}).\n"
        )
    return (
        f"Winner: {best.method}, converged in {best.n_f} function evaluation(s) "
        f"to f={best.f_final:.6g}.\n"
    )


def render_csv(results: Sequence[RunResult], *, objective: str = "") -> str:
    obj = get_objective(objective) if objective else None
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "rank",
            "method",
            "converged",
            "reason",
            "n_f",
            "n_grad",
            "steps",
            "f_final",
            "err_x",
            "x_final",
        ]
    )
    for position, r in enumerate(rank_results(results, objective=objective), start=1):
        err = ""
        if obj is not None and r.x_final:
            dist = sum((a - b) ** 2 for a, b in zip(r.x_final, obj.x_star, strict=True))
            err = f"{dist**0.5:.6g}"
        writer.writerow(
            [
                position,
                r.method,
                int(r.converged),
                r.reason,
                r.n_f,
                r.n_grad,
                len(r.path),
                f"{r.f_final:.6g}" if r.path else "",
                err,
                " ".join(f"{v:.6g}" for v in r.x_final),
            ]
        )
    return buf.getvalue()


def write_csv(path: Path, results: Sequence[RunResult], *, objective: str = "") -> None:
    path.write_text(render_csv(results, objective=objective), encoding="utf-8")
