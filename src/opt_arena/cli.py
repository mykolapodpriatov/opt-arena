"""CLI: race methods and print a table or JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opt_arena.functions import OBJECTIVES
from opt_arena.report import render_csv, write_csv
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
    run_p.add_argument("--max-iter", type=int, default=200)
    run_p.add_argument("--json", action="store_true")
    run_p.add_argument("--csv", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd != "run":
        parser.error("unknown command")

    if args.methods == "auto":
        methods = None
    elif args.methods == "all":
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    budget = Budget(max_iter=args.max_iter)
    results = race(args.function, methods=methods, budget=budget, x0=_parse_x0(args.x0))
    if args.csv is not None:
        write_csv(args.csv, results, objective=args.function)
    if args.json:
        json.dump(
            {"function": args.function, "results": [_result_json(r) for r in results]},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_csv(results, objective=args.function))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
