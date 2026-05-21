#!/usr/bin/env python3
"""Smoke test: parallelism characterisation of the AgentDriver implementations.

Drives both `ClaudeCodeSDKDriver` and `TerminalClaudeDriver` with a trivial
JSON-emitting prompt at sweeping concurrency levels and reports success rate
plus latency per level. Used to validate that the spec's documented
`max_parallelism` defaults (SDK=8, terminal=2) reflect reality on this host.

The prompt is intentionally tiny ("emit `{\"hello\": \"world\"}` and stop") —
this exercises the transport / session-store layer, not the model. Costs
scale with the parallelism levels exercised (default `[1, 2, 4, 8]` => 15
calls per driver; ~$0.03 total on Sonnet at typical short-prompt sizes).

Auto-skips drivers whose dependency is missing:
  - `ClaudeCodeSDKDriver` skipped when `ANTHROPIC_API_KEY` is unset OR when
    `claude-agent-sdk` is not importable.
  - `TerminalClaudeDriver` skipped when the `claude` CLI is not on PATH.

Final line follows the cube-harness smoke contract:
    SMOKE OK: investigator_drivers     (every available driver completed >=1 run at level=1)
    SMOKE FAIL: investigator_drivers   (an available driver failed every call at level=1)
    SMOKE SKIP: investigator_drivers   (no driver was available to test)

Usage:
    uv run scripts/smoke/investigator_drivers.py
    uv run scripts/smoke/investigator_drivers.py --levels 1,2,4,8,16
    uv run scripts/smoke/investigator_drivers.py --skip-terminal
    uv run scripts/smoke/investigator_drivers.py --model claude-haiku-4-5

Reference baselines (Apple M-series macOS, claude-haiku-4-5, ~50/20 token
sizes, recorded 2026-05-13). Treat as orientation, not regression bars —
results vary with API health, network, and concurrent local processes.

  ClaudeCodeSDKDriver (advisory max_parallelism=8):
    level=1    1/1   ok    3.0s wall    3.0s avg
    level=2    2/2   ok    3.3s wall    3.2s avg
    level=3    3/3   ok    4.2s wall    4.0s avg
    level=4    4/4   ok    3.4s wall    2.9s avg
    level=8    8/8   ok    4.6s wall    3.8s avg
    level=16  16/16  ok    8.2s wall    5.7s avg
    level=24  24/24  ok   10.0s wall    8.3s avg
    level=32  32/32  ok   10.8s wall    9.2s avg
    -> Clean through 32. Latency ~3x as parallelism scales 1 -> 32.

  TerminalClaudeDriver (advisory max_parallelism=2):
    level=1    1/1   ok    6.0s wall    6.0s avg
    level=2    2/2   ok    5.4s wall    4.1s avg
    level=3    3/3   ok    3.8s wall    3.5s avg
    level=4    3/4   FAIL  11.6s wall   5.8s avg  (1 stochastic prose-instead-of-JSON)
    level=8    8/8   ok    6.0s wall    4.7s avg
    level=16  16/16  ok    8.9s wall    5.6s avg
    level=24  24/24  ok   10.5s wall    6.9s avg
    level=32  31/32  FAIL  18.8s wall   8.0s avg  (1 `claude -p` exit code 1)
    -> Clean through 24. Real CLI crashes appear around 32.

Takeaways:
  - Both drivers handle far more parallelism on a quiet host than their
    advisory `max_parallelism` defaults suggest. Defaults reflect the
    threshold at which `investigate_experiment` starts clamping with a log line,
    not the hard breaking point.
  - The terminal driver's failure mode at high parallelism is a CLI exit
    crash (session-store contention), not a model-output issue.
  - Stochastic content failures on small prompts can bias single-trial
    runs at any level. Use the levels for a curve, not a hard threshold.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from cube_harness.analyze.investigator.agent_driver import (
    AgentDriver,
    ClaudeCodeSDKDriver,
    DriverResult,
    TerminalClaudeDriver,
)

# Trivial prompt: tiny token footprint, deterministic output, no tools needed.
SYSTEM_PROMPT = "You output a single JSON object and nothing else."
USER_PROMPT = 'Reply with this exact JSON object and nothing else: {"hello": "world"}'

# Empty allowed_tools — no tool calls needed; minimises surface for parallel
# session-store contention on the terminal driver.
EMPTY_ALLOWED_TOOLS: tuple[str, ...] = ()


@dataclass
class CallOutcome:
    """One driver invocation."""

    success: bool
    duration_s: float
    error: str | None = None


@dataclass
class LevelResult:
    """Results for one parallelism level."""

    level: int
    outcomes: list[CallOutcome]
    wall_time_s: float

    @property
    def n_success(self) -> int:
        return sum(1 for o in self.outcomes if o.success)

    @property
    def n_fail(self) -> int:
        return len(self.outcomes) - self.n_success

    @property
    def success_latencies(self) -> list[float]:
        return [o.duration_s for o in self.outcomes if o.success]


@dataclass
class DriverReport:
    """Per-driver smoke output."""

    name: str
    skipped: str | None = None
    level_results: list[LevelResult] = field(default_factory=list)


async def _one_call(driver: AgentDriver, model: str, cwd: Path) -> CallOutcome:
    start = time.time()
    try:
        result: DriverResult = await driver.run(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            cwd=cwd,
            additional_dirs=[],
            model=model,
            allowed_tools=EMPTY_ALLOWED_TOOLS,
            verbose=False,
            trace_mode="off",
        )
    except Exception as e:
        # Truncate the error so the final report stays scannable.
        return CallOutcome(success=False, duration_s=time.time() - start, error=type(e).__name__ + ": " + str(e)[:160])
    duration = time.time() - start
    text = (result.output_text or "").strip()
    if "hello" not in text or "world" not in text:
        # Log the full text — content-mismatch failures are diagnostic gold.
        # Empty responses, prose refusals, and partial JSON each look very
        # different and the smoke is the place where you actually want them.
        if not text:
            detail = "<empty output>"
        else:
            detail = repr(text)
        return CallOutcome(
            success=False,
            duration_s=duration,
            error=f"output missing expected substrings: {detail}",
        )
    return CallOutcome(success=True, duration_s=duration)


async def _run_level(driver: AgentDriver, model: str, cwd: Path, level: int) -> LevelResult:
    start = time.time()
    outcomes = await asyncio.gather(*[_one_call(driver, model, cwd) for _ in range(level)])
    return LevelResult(level=level, outcomes=list(outcomes), wall_time_s=time.time() - start)


async def _exercise_driver(
    driver: AgentDriver, model: str, cwd: Path, levels: list[int], cooldown_s: float
) -> list[LevelResult]:
    results: list[LevelResult] = []
    for level in levels:
        typer.echo(f"  level={level} ...", nl=False)
        out = await _run_level(driver, model, cwd, level)
        avg = statistics.mean(out.success_latencies) if out.success_latencies else float("nan")
        typer.echo(f" {out.n_success}/{out.level} ok, wall={out.wall_time_s:.1f}s, avg-success={avg:.1f}s")
        results.append(out)
        # Brief cooldown so sessions / rate-limits reset between levels.
        if cooldown_s > 0 and level != levels[-1]:
            await asyncio.sleep(cooldown_s)
    return results


def _check_sdk_driver() -> tuple[ClaudeCodeSDKDriver | None, str | None]:
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return None, "claude-agent-sdk package not importable"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set"
    return ClaudeCodeSDKDriver(), None


def _check_terminal_driver() -> tuple[TerminalClaudeDriver | None, str | None]:
    if shutil.which("claude") is None:
        return None, "`claude` CLI not on PATH"
    return TerminalClaudeDriver(), None


def _format_report(report: DriverReport) -> str:
    if report.skipped:
        return f"  [SKIP] {report.skipped}"
    lines = ["  level | n_ok/n  wall(s)  avg-lat(s)  notes"]
    lines.append("  ------+-------------------------------------------")
    for lr in report.level_results:
        avg = statistics.mean(lr.success_latencies) if lr.success_latencies else float("nan")
        notes = f" {lr.n_fail} FAIL" if lr.n_fail else ""
        lines.append(f"  {lr.level:>5} | {lr.n_success:>3}/{lr.level:<3} {lr.wall_time_s:>6.1f}    {avg:>6.1f}{notes}")
    # Print every failure in full beneath the table — content-mismatch text
    # and CLI exit messages are the diagnostic payload of this smoke.
    failure_lines: list[str] = []
    for lr in report.level_results:
        for i, outcome in enumerate(lr.outcomes):
            if not outcome.success:
                failure_lines.append(f"  level={lr.level} call#{i}: {outcome.error}")
    if failure_lines:
        lines.append("")
        lines.append("  Failures:")
        lines.extend(failure_lines)
    return "\n".join(lines)


def _parse_levels(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError as e:
            raise typer.BadParameter(f"--levels must be a comma-separated list of integers, got {part!r}") from e
        if n < 1:
            raise typer.BadParameter(f"--levels values must be >= 1, got {n}")
        out.append(n)
    if not out:
        raise typer.BadParameter("--levels must contain at least one value")
    return out


def main(
    levels: Annotated[
        str,
        typer.Option(help="Comma-separated parallelism levels to exercise."),
    ] = "1,2,4,8",
    model: Annotated[str, typer.Option(help="Model identifier passed to both drivers.")] = "claude-haiku-4-5",
    skip_sdk: Annotated[bool, typer.Option(help="Skip the SDK driver run.")] = False,
    skip_terminal: Annotated[bool, typer.Option(help="Skip the terminal driver run.")] = False,
    cooldown: Annotated[float, typer.Option(help="Seconds between levels (rate-limit / session reset).")] = 2.0,
) -> None:
    """Sweep parallelism for both AgentDriver implementations and report behaviour."""
    parsed_levels = _parse_levels(levels)
    cwd = Path.cwd().resolve()

    typer.echo(f"Driver parallelism smoke — model={model} levels={parsed_levels} cwd={cwd}")
    typer.echo()

    reports: list[DriverReport] = []

    if skip_sdk:
        reports.append(DriverReport(name="ClaudeCodeSDKDriver", skipped="--skip-sdk"))
    else:
        sdk_driver, skip_reason = _check_sdk_driver()
        if sdk_driver is None:
            reports.append(DriverReport(name="ClaudeCodeSDKDriver", skipped=skip_reason or "unavailable"))
        else:
            typer.echo(f"ClaudeCodeSDKDriver (max_parallelism advisory={sdk_driver.max_parallelism})")
            level_results = asyncio.run(_exercise_driver(sdk_driver, model, cwd, parsed_levels, cooldown))
            reports.append(DriverReport(name="ClaudeCodeSDKDriver", level_results=level_results))
            typer.echo()

    if skip_terminal:
        reports.append(DriverReport(name="TerminalClaudeDriver", skipped="--skip-terminal"))
    else:
        terminal_driver, skip_reason = _check_terminal_driver()
        if terminal_driver is None:
            reports.append(DriverReport(name="TerminalClaudeDriver", skipped=skip_reason or "unavailable"))
        else:
            typer.echo(f"TerminalClaudeDriver (max_parallelism advisory={terminal_driver.max_parallelism})")
            level_results = asyncio.run(_exercise_driver(terminal_driver, model, cwd, parsed_levels, cooldown))
            reports.append(DriverReport(name="TerminalClaudeDriver", level_results=level_results))
            typer.echo()

    typer.echo("Report:")
    for r in reports:
        typer.echo(f"\n{r.name}:")
        typer.echo(_format_report(r))

    available = [r for r in reports if not r.skipped]
    if not available:
        typer.echo()
        typer.echo("No driver available to test — set ANTHROPIC_API_KEY or install the `claude` CLI.")
        typer.echo("SMOKE SKIP: investigator_drivers")
        raise typer.Exit(2)

    # SMOKE FAIL when an available driver failed every call at level=1 (the
    # baseline). Higher levels degrading is informational, not a failure — the
    # whole point of the smoke is to discover the degradation curve.
    failures = []
    for r in available:
        baseline = next((lr for lr in r.level_results if lr.level == 1), None)
        if baseline is not None and baseline.n_success == 0:
            failures.append(r.name)

    typer.echo()
    if failures:
        typer.echo(f"Drivers failed at baseline level=1: {', '.join(failures)}")
        typer.echo("SMOKE FAIL: investigator_drivers")
        raise typer.Exit(1)
    typer.echo("SMOKE OK: investigator_drivers")
    raise typer.Exit(0)


if __name__ == "__main__":
    typer.run(main)
