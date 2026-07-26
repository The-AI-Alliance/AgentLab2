#!/usr/bin/env python3
"""wrapping_effort.py — how much code does wrapping a benchmark as a CUBE take?

The paper claims wrapping collapses to "a handful of typed classes plus
metadata". Reviewers asked for that claim to be quantified and compared across
frameworks. This script measures it directly from the corpus.

Counted per CUBE, in non-blank non-comment lines:

  authored   code written by the CUBE author: task, benchmark, tool, configs,
             debug suite. This is the wrapping cost.
  vendored   upstream evaluator source copied in verbatim (see parity_audit.py).
             Not wrapping effort: it is the original benchmark's own scoring
             code, and its provenance is audited separately.
  debug      the LLM-free debug suite the registry's compliance check runs.
             Broken out because it is the one artefact the standard *requires*
             beyond the wrapper itself.

Usage
-----
    scripts/paper/wrapping_effort.py
    scripts/paper/wrapping_effort.py --latex
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
CUBES = REPO_ROOT / "cubes"

# Directories holding upstream evaluator source rather than authored glue.
VENDORED_DIRS = ("vm_backend", "_vendor", "third_party")

# Corpus order follows the paper's Table 5 (registry at submission).
CORPUS: tuple[tuple[str, str], ...] = (
    ("WorkArena", "workarena"),
    ("WebArena-Verified", "webarena-verified"),
    ("MiniWoB", "miniwob"),
    ("BrowseComp", "browsercomp"),
    ("OSWorld", "osworld-cube"),
    ("Windows Agent Arena", "windows-agent-arena-cube"),
    ("SWE-bench Verified", "swebench-verified-cube"),
    ("SWE-bench Live", "swebench-live-cube"),
    ("Terminal-Bench", "terminalbench2-cube"),
)


@dataclass
class Effort:
    """Line counts for one CUBE package."""

    cube: str
    package_dir: str
    authored: int = 0
    vendored: int = 0
    debug: int = 0
    files: int = 0

    @property
    def authored_excl_debug(self) -> int:
        return self.authored - self.debug


def sloc(path: Path) -> int:
    """Non-blank, non-comment, non-docstring lines of Python."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return 0
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            first = body[0]
            docstring_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    count = 0
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or i in docstring_lines:
            continue
        count += 1
    return count


def measure(cube: str, package_dir: str) -> Effort:
    """Walk one cube package and split its lines into the three buckets."""
    effort = Effort(cube=cube, package_dir=package_dir)
    src = CUBES / package_dir / "src"
    if not src.exists():
        return effort
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        lines = sloc(path)
        effort.files += 1
        if any(part in VENDORED_DIRS for part in path.parts):
            effort.vendored += lines
            continue
        effort.authored += lines
        if path.stem.startswith("debug"):
            effort.debug += lines
    return effort


def main(
    latex: Annotated[bool, typer.Option(help="Emit the LaTeX table instead of plain text.")] = False,
    out: Annotated[Path | None, typer.Option(help="Also write the raw measurements as JSON.")] = None,
) -> None:
    """Measure authored wrapping effort for every CUBE in the paper corpus."""
    efforts = [measure(cube, package_dir) for cube, package_dir in CORPUS]
    measured = [e for e in efforts if e.files]

    if latex:
        print(r"\begin{tabular}{lrrr}")
        print(r"\toprule")
        print(r"\textbf{CUBE} & \textbf{Wrapper SLOC} & \textbf{Debug suite} & \textbf{Vendored evaluator} \\")
        print(r"\midrule")
        for e in measured:
            vendored = f"{e.vendored:,}" if e.vendored else "---"
            print(f"{e.cube} & {e.authored_excl_debug:,} & {e.debug:,} & {vendored} \\\\")
        print(r"\midrule")
        median = sorted(e.authored_excl_debug for e in measured)[len(measured) // 2]
        print(f"\\textbf{{Median}} & \\textbf{{{median:,}}} & & \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print(f"{'CUBE':24s} {'wrapper':>8s} {'debug':>7s} {'vendored':>9s} {'files':>6s}")
        for e in measured:
            print(f"{e.cube:24s} {e.authored_excl_debug:8d} {e.debug:7d} {e.vendored:9d} {e.files:6d}")
        totals = (
            sum(e.authored_excl_debug for e in measured),
            sum(e.debug for e in measured),
            sum(e.vendored for e in measured),
        )
        median = sorted(e.authored_excl_debug for e in measured)[len(measured) // 2]
        print(f"\n{'TOTAL':24s} {totals[0]:8d} {totals[1]:7d} {totals[2]:9d}")
        print(f"median wrapper SLOC per CUBE: {median}")

    if out:
        out.write_text(json.dumps([asdict(e) for e in measured], indent=2))


if __name__ == "__main__":
    typer.run(main)
