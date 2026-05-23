"""Tests for the Auto-CUBE headless driver — pure helpers, no SDK call.

`driver.py` lazy-imports the SDK driver inside `run_session`, so importing this
module (and these helpers) needs neither claude-agent-sdk nor the analyze extras.
"""

from pathlib import Path

import pytest

from cube_harness.auto_cube.driver import (
    AUTO_CUBE_ALLOWED_TOOLS,
    build_kickoff,
    load_system_prompt,
)


def test_load_system_prompt_reads_debug_skill() -> None:
    prompt = load_system_prompt("debug")
    assert len(prompt) > 500
    assert "session" in prompt.lower()


def test_load_system_prompt_unknown_use_case_raises() -> None:
    with pytest.raises(FileNotFoundError, match="No SKILL.md for use case 'nope'"):
        load_system_prompt("nope")


def test_allowlist_can_author_and_run() -> None:
    # Broader than the read-only Investigator set: must author files and run experiments.
    for tool in ("Read", "Write", "Edit", "Bash", "Glob", "Grep"):
        assert tool in AUTO_CUBE_ALLOWED_TOOLS


def test_kickoff_new_session_frames_objective_and_paths() -> None:
    msg = build_kickoff("hinter", "raise the slider score", Path("/x/auto_cube/s1"), resume=False)
    assert "raise the slider score" in msg
    assert "s1" in msg
    assert "Create" in msg
    assert "CH_EXP_DIR" in msg


def test_kickoff_resume_points_at_journal() -> None:
    msg = build_kickoff("hinter", "raise the slider score", Path("/x/auto_cube/s1"), resume=True)
    assert "Resume" in msg
    assert "session.md" in msg
