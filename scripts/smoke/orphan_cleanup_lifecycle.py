#!/usr/bin/env python3
"""Smoke: exercise the L2 orphan-cleanup mechanism in ``_experiment_lifecycle``.

Validates behavior that unit tests with ``MagicMock(signal)`` and synchronous
``time.sleep`` cannot fully cover:

- Real signal delivery between OS processes (subprocess + ``os.kill``)
- Real threading: daemon-thread cleanup polled from main thread
- Real timing: bounded grace timeout, escalation message latency
- The Ctrl+C escalation behavior (2nd Ctrl+C during cleanup forces exit)

Each scenario spawns a child Python process that enters
``_experiment_lifecycle`` with a mocked ``InfraConfig`` whose
``cleanup_stale()`` writes a marker file (so the parent can observe progress
without coupling to subprocess stdout). The parent then optionally sends
signals and waits for exit, then asserts on the markers + exit code +
captured stderr.

Scenarios
=========

1. ``normal_exit``        — body returns cleanly. cleanup runs to completion.
2. ``exception_in_body``  — body raises ``RuntimeError``. cleanup still runs.
3. ``sigint_single``      — body sleeps, parent sends 1 SIGINT. cleanup runs
                            to completion, KeyboardInterrupt propagates.
4. ``sigint_escalation``  — body sleeps + cleanup is slow (5s). Parent sends
                            3 SIGINTs: 1st triggers cleanup, 2nd shows the
                            escalation message, 3rd forces exit.
5. ``cleanup_timeout``    — cleanup hangs (sleeps 30s) but grace timeout is
                            2s. Process must exit within ``grace + slop``.
6. ``sigterm``            — body sleeps, parent sends SIGTERM. The
                            ``_raise_systemexit_on_sigterm`` handler converts
                            SIGTERM → SystemExit, finally runs, cleanup
                            completes.

How to read the output
======================
- **SMOKE OK** — every scenario matched its expected markers + exit code +
  stderr signature. The mechanism behaves as documented.
- **SMOKE FAIL** — at least one scenario diverged. The summary table shows
  which one and what was observed vs. expected. This is the failure mode
  you'd act on.
- **SMOKE SKIP** — the cube_harness package wasn't importable. Run from a
  venv where it is.

Each scenario runs in a few seconds; full smoke ~30-60s total.

Run from cube-harness repo root:

    python scripts/smoke/orphan_cleanup_lifecycle.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

NAME = "orphan_cleanup_lifecycle"

# Generous slack on top of the configured grace timeout for subprocess exit.
# The lifecycle has its own timeout enforcement; we just need to not give up
# before that fires.
SUBPROCESS_EXIT_SLOP_S: float = 10.0

# Wait for the child to enter the lifecycle before signalling. The child writes
# `inner_ready` immediately; we poll for it rather than guessing a sleep.
INNER_READY_TIMEOUT_S: float = 15.0


def banner(status: str, reason: str = "") -> int:
    """Print the SMOKE banner and return the conventional exit code."""
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


# Multi-line Python source that runs in the child process.
#
# Reads scenario config from env vars (so we can spawn the same script with
# different behaviors), enters _experiment_lifecycle with a mock infra whose
# cleanup_stale() writes markers, then either returns / raises / sleeps based
# on SMOKE_MODE. Markers are timestamped files in SMOKE_MARKERS_DIR.
_INNER_HELPER = textwrap.dedent(
    """\
    import os, sys, time
    from pathlib import Path
    from unittest.mock import MagicMock

    import cube_harness.exp_runner as exp_runner_mod
    from cube.resource import InfraConfig
    from cube_harness.exp_runner import _experiment_lifecycle

    timeout = float(os.environ.get("SMOKE_TIMEOUT_S", "30"))
    exp_runner_mod._CLEANUP_GRACE_TIMEOUT_S = timeout

    mode = os.environ.get("SMOKE_MODE", "normal")
    cleanup_delay = float(os.environ.get("SMOKE_CLEANUP_DELAY_S", "0"))
    markers = Path(os.environ["SMOKE_MARKERS_DIR"])
    markers.mkdir(parents=True, exist_ok=True)

    def emit(name: str) -> None:
        (markers / name).write_text(f"{time.time():.3f}")

    infra = MagicMock(spec=InfraConfig)
    def _cleanup(*a, **kw):
        emit("cleanup_started")
        if cleanup_delay > 0:
            time.sleep(cleanup_delay)
        emit("cleanup_completed")
        return []
    infra.cleanup_stale.side_effect = _cleanup

    emit("inner_ready")

    try:
        with _experiment_lifecycle(Path(os.environ.get("SMOKE_EXP_DIR", "/tmp")),
                                   mode="sequential", infra=infra):
            if mode == "sleep":
                # Pass through KeyboardInterrupt naturally
                time.sleep(60)
            elif mode == "raise":
                raise RuntimeError("smoke: test exception in body")
            # mode == "normal": just exit the with block
    except KeyboardInterrupt:
        emit("exit_keyboard_interrupt")
        sys.exit(130)
    except SystemExit as e:
        emit("exit_system_exit")
        raise
    except RuntimeError as e:
        emit("exit_runtime_error")
        sys.exit(2)

    emit("exit_clean")
    """
)


@dataclass
class ScenarioResult:
    """Captured observations from one scenario run."""

    name: str
    expected_markers: list[str]
    forbidden_markers: list[str]
    observed_markers: list[str] = field(default_factory=list)
    expect_stderr_contains: list[str] = field(default_factory=list)
    stderr_seen: str = ""
    expected_exit_code: int | None = None
    actual_exit_code: int | None = None
    wall_clock_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        if self.error:
            return False
        if self.expected_exit_code is not None and self.actual_exit_code != self.expected_exit_code:
            return False
        if set(self.expected_markers) - set(self.observed_markers):
            return False
        if set(self.forbidden_markers) & set(self.observed_markers):
            return False
        for needle in self.expect_stderr_contains:
            if needle not in self.stderr_seen:
                return False
        return True


def _wait_for_marker(markers_dir: Path, name: str, timeout_s: float) -> bool:
    """Poll for a marker file to appear. Returns True if found, False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (markers_dir / name).exists():
            return True
        time.sleep(0.05)
    return False


def _list_markers(markers_dir: Path) -> list[str]:
    """Return marker filenames in the order they were written (by mtime)."""
    if not markers_dir.exists():
        return []
    entries = [(p.stat().st_mtime, p.name) for p in markers_dir.iterdir() if p.is_file()]
    entries.sort()
    return [name for _, name in entries]


def _run_scenario(
    name: str,
    *,
    mode: str = "normal",
    cleanup_delay_s: float = 0.0,
    lifecycle_timeout_s: float = 30.0,
    signals: list[tuple[float, int]] | None = None,
    expected_markers: list[str],
    forbidden_markers: list[str] | None = None,
    expect_stderr_contains: list[str] | None = None,
    expected_exit_code: int | None = None,
    overall_timeout_s: float = 60.0,
) -> ScenarioResult:
    """Run a single scenario.

    Spawns ``python -c <inner_helper>`` with env vars + markers dir, optionally
    sends signals at scheduled offsets from launch, waits for exit, captures
    markers and stderr, returns a ScenarioResult.
    """
    result = ScenarioResult(
        name=name,
        expected_markers=expected_markers,
        forbidden_markers=forbidden_markers or [],
        expect_stderr_contains=expect_stderr_contains or [],
        expected_exit_code=expected_exit_code,
    )
    markers_dir = Path(tempfile.mkdtemp(prefix=f"smoke-{name}-"))
    try:
        env = {
            **os.environ,
            "SMOKE_MODE": mode,
            "SMOKE_CLEANUP_DELAY_S": str(cleanup_delay_s),
            "SMOKE_TIMEOUT_S": str(lifecycle_timeout_s),
            "SMOKE_MARKERS_DIR": str(markers_dir),
            "PYTHONUNBUFFERED": "1",
        }
        proc = subprocess.Popen(
            [sys.executable, "-c", _INNER_HELPER],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,  # so SIGINT from us goes to subprocess only
        )
        t0 = time.time()

        # Wait for child to signal it's inside the lifecycle. Without this we
        # might send SIGINT before the with-block is entered.
        if not _wait_for_marker(markers_dir, "inner_ready", INNER_READY_TIMEOUT_S):
            proc.kill()
            proc.wait()
            result.error = f"child did not become ready within {INNER_READY_TIMEOUT_S}s"
            return result

        # Deliver scheduled signals at their offsets from t0.
        for offset_s, sig in signals or []:
            elapsed = time.time() - t0
            sleep_for = offset_s - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            try:
                os.kill(proc.pid, sig)
            except ProcessLookupError:
                # Child already exited; remaining signals are no-ops.
                break

        try:
            stderr_bytes = proc.communicate(timeout=overall_timeout_s)[1] or b""
        except subprocess.TimeoutExpired:
            proc.kill()
            stderr_bytes = proc.communicate()[1] or b""
            result.error = f"subprocess did not exit within {overall_timeout_s}s"

        result.wall_clock_s = time.time() - t0
        result.actual_exit_code = proc.returncode
        result.stderr_seen = stderr_bytes.decode("utf-8", errors="replace")
        result.observed_markers = _list_markers(markers_dir)
        return result
    finally:
        shutil.rmtree(markers_dir, ignore_errors=True)


def _print_results(results: list[ScenarioResult]) -> None:
    """Print a compact summary table of all scenario observations."""
    print("\n" + "─" * 78)
    print(f"{'scenario':<22} {'verdict':<8} {'wall':>7} {'exit':>5}  markers/notes")
    print("─" * 78)
    for r in results:
        verdict = "OK" if r.ok else "FAIL"
        wall = f"{r.wall_clock_s:.1f}s"
        exit_str = "?" if r.actual_exit_code is None else str(r.actual_exit_code)
        markers_summary = ",".join(r.observed_markers)
        line = f"{r.name:<22} {verdict:<8} {wall:>7} {exit_str:>5}  {markers_summary}"
        print(line)
        if r.error:
            print(f"  {'':<22} error:    {r.error}")
        if not r.ok:
            missing = set(r.expected_markers) - set(r.observed_markers)
            forbidden_seen = set(r.forbidden_markers) & set(r.observed_markers)
            stderr_missing = [s for s in r.expect_stderr_contains if s not in r.stderr_seen]
            if missing:
                print(f"  {'':<22} missing:  {sorted(missing)}")
            if forbidden_seen:
                print(f"  {'':<22} forbidden:{sorted(forbidden_seen)}")
            if stderr_missing:
                print(f"  {'':<22} stderr missing: {stderr_missing}")
            if r.expected_exit_code is not None and r.actual_exit_code != r.expected_exit_code:
                print(f"  {'':<22} exit code: expected {r.expected_exit_code}, got {r.actual_exit_code}")
            stderr_tail = r.stderr_seen.strip().splitlines()[-3:] if r.stderr_seen else []
            if stderr_tail:
                print(f"  {'':<22} stderr tail: {stderr_tail}")
    print("─" * 78)


def main() -> int:
    """Run all scenarios and return SMOKE OK/FAIL/SKIP exit code."""
    if importlib.util.find_spec("cube_harness") is None:
        return banner("SKIP", "cube_harness not installed in this environment")

    print(f"smoke: {NAME} — exercising _experiment_lifecycle signal + cleanup behavior")

    scenarios: list[ScenarioResult] = []

    # --- 1. Normal exit ----------------------------------------------------
    scenarios.append(
        _run_scenario(
            "normal_exit",
            mode="normal",
            expected_markers=["inner_ready", "cleanup_started", "cleanup_completed", "exit_clean"],
            forbidden_markers=["exit_keyboard_interrupt", "exit_system_exit", "exit_runtime_error"],
            expected_exit_code=0,
        )
    )

    # --- 2. Exception in body ---------------------------------------------
    scenarios.append(
        _run_scenario(
            "exception_in_body",
            mode="raise",
            expected_markers=["inner_ready", "cleanup_started", "cleanup_completed", "exit_runtime_error"],
            forbidden_markers=["exit_clean"],
            expected_exit_code=2,
        )
    )

    # --- 3. Single SIGINT --------------------------------------------------
    scenarios.append(
        _run_scenario(
            "sigint_single",
            mode="sleep",
            signals=[(1.0, signal.SIGINT)],
            expected_markers=["inner_ready", "cleanup_started", "cleanup_completed", "exit_keyboard_interrupt"],
            forbidden_markers=["exit_clean"],
            expected_exit_code=130,
        )
    )

    # --- 4. SIGINT escalation (force exit on the 2nd press DURING cleanup) -
    # Cleanup takes 5s; first SIGINT enters cleanup, second SIGINT triggers
    # the escalation message, third sets the force flag and exits.
    scenarios.append(
        _run_scenario(
            "sigint_escalation",
            mode="sleep",
            cleanup_delay_s=5.0,
            lifecycle_timeout_s=30.0,
            signals=[(1.0, signal.SIGINT), (2.0, signal.SIGINT), (3.0, signal.SIGINT)],
            expected_markers=["inner_ready", "cleanup_started", "exit_keyboard_interrupt"],
            forbidden_markers=["cleanup_completed"],
            expect_stderr_contains=["Cleanup in progress", "Forcing exit"],
            expected_exit_code=130,
            overall_timeout_s=20.0,
        )
    )

    # --- 5. Cleanup timeout (no signal, just hung cleanup) -----------------
    # Body returns immediately, cleanup hangs 30s, grace is 2s.
    scenarios.append(
        _run_scenario(
            "cleanup_timeout",
            mode="normal",
            cleanup_delay_s=30.0,
            lifecycle_timeout_s=2.0,
            expected_markers=["inner_ready", "cleanup_started", "exit_clean"],
            forbidden_markers=["cleanup_completed"],
            expected_exit_code=0,
            overall_timeout_s=15.0,
        )
    )

    # --- 6. SIGTERM (orchestrator-style shutdown) --------------------------
    # The SIGTERM handler converts to SystemExit so finally runs.
    scenarios.append(
        _run_scenario(
            "sigterm",
            mode="sleep",
            signals=[(1.0, signal.SIGTERM)],
            expected_markers=["inner_ready", "cleanup_started", "cleanup_completed", "exit_system_exit"],
            forbidden_markers=["exit_clean", "exit_keyboard_interrupt"],
            # SystemExit with non-None code: subprocess returns that code.
            # _raise_systemexit_on_sigterm raises SystemExit(128 + 15) = 143.
            expected_exit_code=143,
        )
    )

    _print_results(scenarios)

    if all(r.ok for r in scenarios):
        return banner("OK", f"all {len(scenarios)} scenarios passed")
    failed = [r.name for r in scenarios if not r.ok]
    return banner("FAIL", f"{len(failed)} scenario(s) diverged: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
