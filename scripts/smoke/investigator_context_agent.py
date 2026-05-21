#!/usr/bin/env python3
"""Smoke test: benchmark-context sub-agent live invocation.

Calls `generate_context_file(experiment_dir, driver=ClaudeCodeSDKDriver())`
against a tiny fixture experiment dir whose `experiment_config.json`
references real importable types (`cube_harness.agents.react.ReactAgentConfig`)
plus a fake benchmark `_type` to exercise the agent's "skip missing"
behaviour.

What this exercises that unit tests cannot:
  - The real Opus-by-default invocation of the sub-agent.
  - The system prompt actually elicits a well-formed ```paths fenced block.
  - `_extract_markdown` accepts whatever Opus emits (raw markdown,
    ```markdown fence, etc.).
  - `validate_context_file` parses the agent's output without raising.
  - Every path the agent listed resolves on disk (no hallucinations).

Cost: roughly $0.10-$0.20 per run on Opus, since the sub-agent typically
runs Bash + Glob calls to verify each candidate path. Use `--model` to
override (haiku is much cheaper but less reliable at structured emission).

Auto-skips when `claude-agent-sdk` is not importable or `ANTHROPIC_API_KEY`
is unset.

Usage:
    uv run scripts/smoke/investigator_context_agent.py
    uv run scripts/smoke/investigator_context_agent.py --model claude-sonnet-4-6
    uv run scripts/smoke/investigator_context_agent.py --keep-temp     # inspect the output

Final line follows the cube-harness smoke contract:
    SMOKE OK: investigator_context_agent
    SMOKE FAIL: investigator_context_agent
    SMOKE SKIP: investigator_context_agent
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.investigator.agent_driver import ClaudeCodeSDKDriver
from cube_harness.analyze.investigator.benchmark_context_agent import (
    DEFAULT_CONTEXT_MODEL,
    generate_context_file,
)
from cube_harness.analyze.investigator.context import validate_context_file


def _build_fixture(root: Path) -> Path:
    """Create `<root>/exp/` with a realistic `experiment_config.json`.

    The agent _type is real (`ReactAgentConfig` is shipped with cube-harness)
    so the sub-agent can resolve at least one cube-side package. The benchmark
    _type points at a deliberately-missing module — that exercises the agent's
    "skip rather than hallucinate" instruction.
    """
    exp = root / "exp"
    exp.mkdir(parents=True)
    (exp / "experiment_config.json").write_text(
        json.dumps(
            {
                "name": "investigator_context_agent_smoke",
                "agent_config": {"_type": "cube_harness.agents.react.ReactAgentConfig"},
                # Intentionally missing: tests "skip missing" path.
                "benchmark_config": {"_type": "nonexistent_smoke_benchmark.MyBenchmarkConfig"},
            }
        )
    )
    return exp


def _check_driver() -> tuple[ClaudeCodeSDKDriver | None, str | None]:
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return None, "claude-agent-sdk package not importable"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set"
    return ClaudeCodeSDKDriver(), None


def _verify(context_path: Path, resolved: dict) -> list[tuple[bool, str]]:
    """Sanity-check the emitted context file."""
    checks: list[tuple[bool, str]] = []

    # 1. File exists and is non-trivial.
    checks.append(
        (context_path.exists() and context_path.stat().st_size > 50, f"context file written ({context_path})")
    )

    # 2. At least one path was emitted.
    checks.append((len(resolved) >= 1, f"resolved at least 1 path ({len(resolved)} total)"))

    # 3. Every path resolves on disk — `validate_context_file` already raises
    # when one doesn't, so reaching here means they all exist. Make the check
    # explicit anyway for the report.
    all_exist = all(p.exists() for p in resolved.values())
    checks.append((all_exist, "every listed path exists locally"))

    # 4. cube_harness and cube source roots should appear — the system prompt
    # mandates them. Match by substring on the resolved values rather than the
    # name (the agent picks the name; we can't dictate it).
    paths_str = " ".join(str(p) for p in resolved.values())
    checks.append(("cube_harness" in paths_str, "cube_harness source root referenced"))
    checks.append(("cube" in paths_str, "cube (cube-standard) source root referenced"))

    # 5. The missing benchmark _type should NOT have produced a fabricated path
    # — every path is real, so if any entry's basename contains
    # 'nonexistent_smoke_benchmark', the agent hallucinated.
    no_hallucination = "nonexistent_smoke_benchmark" not in paths_str
    checks.append((no_hallucination, "no hallucinated benchmark path"))

    return checks


def main(
    model: Annotated[
        str,
        typer.Option(help="Model for the benchmark-context sub-agent. Opus by default."),
    ] = DEFAULT_CONTEXT_MODEL,
    keep_temp: Annotated[bool, typer.Option(help="Leave the fixture for inspection.")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run the benchmark-context sub-agent live and verify its output."""
    typer.echo(f"investigator_context_agent — model={model}")

    driver, skip_reason = _check_driver()
    if driver is None:
        typer.echo(f"Driver unavailable: {skip_reason}")
        typer.echo("SMOKE SKIP: investigator_context_agent")
        raise typer.Exit(2)

    tmp_root = Path(tempfile.mkdtemp(prefix="investigator_context_agent_smoke_"))
    typer.echo(f"Fixture root: {tmp_root}")

    try:
        exp_dir = _build_fixture(tmp_root)
        typer.echo(f"Built fixture experiment at {exp_dir}")
        typer.echo("Invoking benchmark-context sub-agent ...")

        context_path = asyncio.run(
            generate_context_file(exp_dir, driver=driver, model=model, verbose=verbose),
        )
        typer.echo(f"Sub-agent wrote {context_path}")
        typer.echo()
        typer.echo("=== investigation_context.md content ===")
        typer.echo(context_path.read_text())
        typer.echo("================================")
        typer.echo()

        resolved = validate_context_file(context_path)
    except Exception as e:
        typer.echo(f"Smoke raised {type(e).__name__}: {e}")
        if not keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        typer.echo("SMOKE FAIL: investigator_context_agent")
        raise typer.Exit(1) from e

    typer.echo("Resolved paths:")
    for name, p in resolved.items():
        typer.echo(f"  {name}: {p}")
    typer.echo()

    checks = _verify(context_path, resolved)
    passed = sum(1 for ok, _ in checks if ok)
    total = len(checks)
    for ok, label in checks:
        marker = "✓" if ok else "✗"
        typer.echo(f"  {marker} {label}")

    if not keep_temp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        typer.echo()
        typer.echo("(fixture cleaned up — pass --keep-temp to inspect next time)")
    else:
        typer.echo()
        typer.echo(f"Fixture preserved at {tmp_root}")

    typer.echo()
    typer.echo(f"Summary: {passed}/{total} checks passed.")
    if passed == total:
        typer.echo("SMOKE OK: investigator_context_agent")
        raise typer.Exit(0)
    typer.echo("SMOKE FAIL: investigator_context_agent")
    raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
