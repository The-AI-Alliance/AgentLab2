"""Pins the verifier-timeout invariant in `TerminalBench2Task.evaluate`.

The verifier is trusted infra and must run with its full `max_test_timeout_sec`
(up to 7200s for some tasks). The agent-facing `bash` clamps every call to the
tool's `max_timeout` (900s abuse guard) and truncates output — which silently
killed the verifier at 900s for the 20 tasks whose `max_test_timeout_sec` exceeds
900 (e.g. reshard-c4-data=3600, sam-cell-seg=7200). `evaluate` must run test.sh
via `bash_unlimited`, which skips the clamp and truncation.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from cube.tools.terminal import ContainerTerminalTool, TerminalToolConfig

from terminalbench2_cube.task import TerminalBench2Task


def _tool_with_cap() -> tuple[ContainerTerminalTool, MagicMock]:
    container = MagicMock()
    container.exec.return_value = MagicMock(stdout="ok", stderr="", exit_code=0, duration_seconds=1.0)
    tool = ContainerTerminalTool(config=TerminalToolConfig(working_dir="/app", max_timeout=900), container=container)
    return tool, container


def test_bash_clamps_but_bash_unlimited_does_not() -> None:
    """The behavioral property the fix relies on: bash clamps to max_timeout (900),
    bash_unlimited passes the full timeout through to the container."""
    tool, container = _tool_with_cap()

    tool.bash("echo x", timeout=3600)
    assert container.exec.call_args.kwargs["timeout"] == 900  # agent-facing: clamped

    container.exec.reset_mock()
    tool.bash_unlimited("echo x", timeout=3600)
    assert container.exec.call_args.kwargs["timeout"] == 3600  # verifier-facing: uncapped


def test_evaluate_runs_verifier_via_bash_unlimited() -> None:
    """Regression guard: evaluate() must run the test.sh verifier via bash_unlimited
    (not the clamped/truncated agent-facing bash)."""
    src = inspect.getsource(TerminalBench2Task.evaluate)
    # the verifier RUN (bash <tests>/test.sh) goes through bash_unlimited...
    assert "bash_unlimited(" in src
    # ...and is NOT run through the clamped agent-facing bash. (chmod/mkdir on the
    # test path are quick commands and may legitimately use bash — exclude them.)
    for line in src.splitlines():
        runs_testsh = "bash " in line and "test.sh" in line  # actually executing the script
        if runs_testsh and "self.tool.bash(" in line and "chmod" not in line and "mkdir" not in line:
            raise AssertionError(f"verifier test.sh must use bash_unlimited, not bash: {line.strip()}")
