"""Regression test for `TerminalBench2Task._build_tool` best-effort git setup.

Pins the invariant that the relocate-fallback's `extra_setup` returns exit 0
regardless of whether `git` is on PATH, so task images that omit git (nginx,
sqlite, configure-git-webserver) don't abort container setup on non-root
infras like EAI toolkit (uid 13011, read-only /app). See F4 in
`2026-05-20_tbench2-infra-model-matrix` session journal.
"""

from __future__ import annotations

import os
import re
import subprocess

from terminalbench2_cube import task as task_module


def _extract_extra_setup() -> str:
    """Read the `extra_setup=(...)` literal from task._build_tool."""
    src = open(task_module.__file__, encoding="utf-8").read()
    match = re.search(r"extra_setup=\(\s*((?:\".*?\"\s*)+)\s*\),", src, re.DOTALL)
    assert match, "Could not find extra_setup literal in task.py"
    # Concatenate the consecutive string literals.
    return "".join(re.findall(r'"([^"]*)"', match.group(1)))


def test_extra_setup_runs_clean_when_git_present() -> None:
    """With git on PATH, the chain still completes successfully (no false fail)."""
    snippet = _extract_extra_setup()
    result = subprocess.run(
        ["sh", "-c", snippet],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    assert result.returncode == 0, (
        f"extra_setup must exit 0 when git is present. got {result.returncode}, stderr={result.stderr!r}"
    )


def test_extra_setup_runs_clean_when_git_absent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """With git absent from PATH, the chain must exit 0 — the bug-fixing invariant.

    Pre-fix: `git: not found` exit 127 propagated to relocate_if_readonly →
    ContainerExecError → task setup aborted. Post-fix: `command -v git` gate
    skips the chain and `|| true` neutralises the non-zero from the guard.
    """
    snippet = _extract_extra_setup()
    # Build a PATH that has core shell utils (cp, true, etc.) but no `git`.
    # Symlink the bins we need into a fresh dir; omit git deliberately.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("sh", "cp", "true", "cat", "echo"):
        src = next((p for p in ("/bin", "/usr/bin") if os.path.exists(f"{p}/{name}")), None)
        if src is not None:
            os.symlink(f"{src}/{name}", bin_dir / name)
    result = subprocess.run(
        ["/bin/sh", "-c", snippet],
        capture_output=True,
        text=True,
        env={"PATH": str(bin_dir)},
        check=False,
    )
    assert result.returncode == 0, (
        f"extra_setup must exit 0 when git is absent. got {result.returncode}, stderr={result.stderr!r}"
    )


def test_cp_failure_is_NOT_masked_by_extra_setup() -> None:
    """The relocate-fallback's `cp -a … && <extra_setup>` must propagate a cp
    failure. auto-fix(176) raises ContainerExecError on cp failure; this PR's
    `|| true` must apply ONLY to the git chain, never to the cp step.

    Pre-F8-fix regression: `cp_fail && (… || true)` → the outer `||` neutralised
    the whole chain, so `relocate_if_readonly` returned a working_dir that
    didn't exist. Surfaced as `prove-plus-comm` task health-check failures
    on daytona (image with read-only /app + uncopyable file).
    """
    snippet = _extract_extra_setup()
    # Simulate the full relocate command shape: `cp_failure && <extra_setup>`.
    # `false` stands in for a failing `cp`. The whole shell expression must
    # propagate the non-zero from `false`, not be masked by `|| true` inside
    # extra_setup.
    full_cmd = f"false && {snippet}"
    result = subprocess.run(
        ["sh", "-c", full_cmd],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    assert result.returncode != 0, (
        "extra_setup must NOT mask a cp failure when chained via "
        f"`cp && <extra_setup>`. got returncode={result.returncode}, stderr={result.stderr!r}"
    )
