"""Storage event-file layout tests — save_event / load_event,
crash-safe save_metadata roundtrip.

The historical `Trajectory(events=[...])` round-trip tests have been
moved to `tests/test_episode_view.py`, which exercises the canonical
`TrajectoryMetadata + TrajectoryView` path. This file now only covers the
low-level `save_event` / `load_event` storage methods directly.
"""

from pathlib import Path

import pytest
from cube.core import Action, EnvironmentOutput, Observation

from cube_harness.core import (
    AgentEvent,
    EvaluationEvent,
    ToolCallEvent,
    TrajectoryEvent,
    TrajectoryMetadata,
)
from cube_harness.storage import EVENTS_DIR, FileStorage


def _agent_event(thoughts: str = "t") -> AgentEvent:
    return AgentEvent(actions=[Action(id="a-1", name="foo", arguments={})], thoughts=thoughts)


def _tool_call_event(parent_id: str) -> ToolCallEvent:
    return ToolCallEvent(
        parent_event_id=parent_id,
        action_id="a-1",
        output=EnvironmentOutput(obs=Observation(), reward=0.5, done=False, info={}),
        turn_id=parent_id,
    )


def _eval_event(reward: float = 1.0) -> EvaluationEvent:
    return EvaluationEvent(reward=reward, info={"score": reward})


def _prime(storage: FileStorage, traj_id: str) -> None:
    """Write a stub TrajectoryMetadata so the episode directory exists —
    save_event requires the directory and we want to test save_event
    without exercising save_trajectory."""
    storage.save_metadata(TrajectoryMetadata(id=traj_id))


# ---------------------------------------------------------------------------
# save_event / load_event
# ---------------------------------------------------------------------------


def test_save_and_load_event_round_trip(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    _prime(storage, "t")

    parent = _agent_event()
    te = TrajectoryEvent(output=parent, start_time=0.0, end_time=0.1)
    storage.save_event(te, "t")

    # Loaded event round-trips through msgpack+zstd serialization.
    loaded = storage.load_event("t", 0)
    assert isinstance(loaded.output, AgentEvent)
    assert loaded.output.thoughts == parent.thoughts


def test_save_event_creates_events_dir_lazily(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    _prime(storage, "t")
    ep_dir = tmp_path / "episodes" / "t"
    assert not (ep_dir / EVENTS_DIR).exists()
    storage.save_event(TrajectoryEvent(output=_eval_event(), start_time=0.0, end_time=0.0), "t")
    assert (ep_dir / EVENTS_DIR).exists()


def test_save_event_filename_carries_kind(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    _prime(storage, "t")
    storage.save_event(TrajectoryEvent(output=_agent_event()), "t")
    storage.save_event(TrajectoryEvent(output=_tool_call_event("p")), "t")
    storage.save_event(TrajectoryEvent(output=_eval_event()), "t")
    files = sorted((tmp_path / "episodes" / "t" / EVENTS_DIR).iterdir())
    names = [f.name for f in files]
    assert "000_agent.msgpack.zst" in names
    assert "001_tool_call.msgpack.zst" in names
    assert "002_eval.msgpack.zst" in names


def test_save_event_requires_episode_dir(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    with pytest.raises(ValueError):
        storage.save_event(TrajectoryEvent(output=_eval_event()), "missing")


def test_load_event_missing_raises(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path)
    _prime(storage, "t")
    with pytest.raises(FileNotFoundError):
        storage.load_event("t", 99)
