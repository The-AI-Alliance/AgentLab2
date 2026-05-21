"""Regression test for the `/opt/cube/uv` version guard in `_ensure_uv_preinstalled`.

Pins the invariant: when `/opt/cube/uv` is present but missing the
`uvx --with` flag (the syntax tbench2's test.sh depends on), the
fast-path copy is skipped and the function falls through to the
pip-install path. See F6 in 2026-05-20_tbench2-infra-model-matrix
session journal.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from terminalbench2_cube.task import TerminalBench2Task


def _make_task(bash_responses: list[str]) -> TerminalBench2Task:
    """Build a minimal `TerminalBench2Task` for unit-testing helpers without
    standing up a real container.

    Bypasses Pydantic init entirely (`__new__` + private-attr injection); the
    only behavior `_ensure_uv_preinstalled` needs is `self.tool.bash(...)`,
    which we wire to a MagicMock that yields responses in order.
    """
    task = TerminalBench2Task.__new__(TerminalBench2Task)
    fake_tool = MagicMock()
    fake_tool.bash.side_effect = list(bash_responses)
    # Bypass Pydantic field validators on private attrs.
    object.__setattr__(task, "_tool", fake_tool)
    object.__setattr__(task, "_container", None)
    return task


def test_fast_path_taken_when_uv_supports_with() -> None:
    """A fresh `/opt/cube/uv` (supports `--with`) → fast-path copy, no pip-install."""
    task = _make_task(
        [
            "MISSING",  # marker probe: /tmp/fakehome/.local/bin/uv absent
            "YES",  # assets probe: /opt/cube/uv + uvx present
            "OK",  # version probe: uvx --help has ' --with '
            "",  # the copy command output
        ]
    )
    task._ensure_uv_preinstalled()
    calls = [c.args[0] for c in task._tool.bash.call_args_list]
    assert any("cp /opt/cube/uv /opt/cube/uvx" in c for c in calls), (
        "Fast path must copy from /opt/cube/uv when version supports --with"
    )
    assert not any("python3 -m pip install" in c or "pip install --quiet --target" in c for c in calls), (
        "pip-install path must NOT be entered when fast path succeeds"
    )


def test_fast_path_skipped_when_uv_lacks_with() -> None:
    """A stale `/opt/cube/uv` (no `--with`) → skip fast path, fall through."""
    task = _make_task(
        [
            "MISSING",  # marker probe: target uv absent
            "YES",  # assets probe: /opt/cube/uv present
            "OLD",  # version probe: uvx --help has NO ' --with '
            "HAS_PYTHON\nPython 3.11.0",  # has_python check (root path)
            "missing",  # use_extracted probe
            "",  # actual pip install cmd output
        ]
    )
    task._ensure_uv_preinstalled()
    calls = [c.args[0] for c in task._tool.bash.call_args_list]
    # The probe ran but the copy step was NOT invoked.
    assert any(" --with " in c for c in calls), "Version probe must have run"
    copy_calls = [c for c in calls if "cp /opt/cube/uv /opt/cube/uvx" in c]
    assert not copy_calls, f"Fast path must NOT copy when /opt/cube/uv lacks --with; got: {copy_calls!r}"
    assert any("pip install" in c and "uv" in c for c in calls), "Fall-through must invoke a pip-install command for uv"
