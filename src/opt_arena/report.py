"""CSV comparison of a race."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from opt_arena.functions import get_objective
from opt_arena.types import RunResult


def render_csv(results: Sequence[RunResult], *, objective: str = "") -> str:
    obj = get_objective(objective) if objective else None
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
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
    for r in results:
        err = ""
        if obj is not None and r.x_final:
            dist = sum((a - b) ** 2 for a, b in zip(r.x_final, obj.x_star, strict=True))
            err = f"{dist**0.5:.6g}"
        writer.writerow(
            [
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
