from __future__ import annotations

import json
from pathlib import Path

from opt_arena.cli import main
from opt_arena.report import render_csv
from opt_arena.run import race


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
    assert lines[0].startswith("method,")
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
