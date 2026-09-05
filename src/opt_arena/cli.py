"""CLI: race methods and print a table or JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opt_arena.functions import OBJECTIVES
from opt_arena.report import rank_results, render_csv, render_verdict, write_csv
from opt_arena.run import ALL_METHODS, race
from opt_arena.types import Budget, RunResult


def _parse_x0(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    return [float(part) for part in raw.split(",") if part.strip()]


def _result_json(r: RunResult) -> dict[str, object]:
    return {
        "method": r.method,
        "converged": r.converged,
        "reason": r.reason,
        "n_f": r.n_f,
        "n_grad": r.n_grad,
        "x_final": r.x_final,
        "f_final": r.f_final,
        "path": r.path,
        "f_path": r.f_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opt-arena")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="race methods on one function")
    run_p.add_argument("--function", required=True, choices=sorted(OBJECTIVES))
    run_p.add_argument(
        "--methods",
        default="auto",
        help="comma list or 'auto' / 'all'",
    )
    run_p.add_argument("--x0", default=None, help="comma-separated start, e.g. -1.2,1")
    # Budget defaults come from the dataclass rather than being restated here,
    # so the CLI and the library cannot drift apart on what "default" means.
    _defaults = Budget()
    run_p.add_argument("--max-iter", type=int, default=_defaults.max_iter)
    run_p.add_argument(
        "--max-f",
        type=int,
        default=_defaults.max_f,
        help=(
            "cap on function evaluations. This is the fair axis for comparing "
            "method classes: one CG iteration costs a whole line search."
        ),
    )
    run_p.add_argument(
        "--max-grad", type=int, default=_defaults.max_grad, help="cap on gradient evaluations"
    )
    run_p.add_argument(
        "--tol-x", type=float, default=_defaults.tol_x, help="step-size stopping tolerance"
    )
    run_p.add_argument(
        "--tol-f", type=float, default=_defaults.tol_f, help="objective-decrease stopping tolerance"
    )
    run_p.add_argument(
        "--tol-g", type=float, default=_defaults.tol_g, help="gradient-norm stopping tolerance"
    )
    run_p.add_argument("--json", action="store_true")
    run_p.add_argument("--csv", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd != "run":
        parser.error("unknown command")

    for flag, value in (
        ("--max-iter", args.max_iter),
        ("--max-f", args.max_f),
        ("--max-grad", args.max_grad),
    ):
        if value < 1:
            parser.error(f"{flag} must be at least 1, got {value}")
    for flag, value in (("--tol-x", args.tol_x), ("--tol-f", args.tol_f), ("--tol-g", args.tol_g)):
        if value <= 0:
            parser.error(f"{flag} must be positive, got {value}")

    if args.methods == "auto":
        methods = None
    elif args.methods == "all":
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    budget = Budget(
        max_iter=args.max_iter,
        max_f=args.max_f,
        max_grad=args.max_grad,
        tol_x=args.tol_x,
        tol_f=args.tol_f,
        tol_g=args.tol_g,
    )
    results = race(args.function, methods=methods, budget=budget, x0=_parse_x0(args.x0))
    ranked = rank_results(results, objective=args.function)
    if args.csv is not None:
        # A data file gets the rank column but not the prose line.
        write_csv(args.csv, results, objective=args.function)
    if args.json:
        # The consumer gets the same ordering rather than having to re-derive it.
        json.dump(
            {
                "function": args.function,
                "results": [{**_result_json(r), "rank": i} for i, r in enumerate(ranked, start=1)],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_csv(results, objective=args.function))
        sys.stdout.write(render_verdict(results, objective=args.function))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
