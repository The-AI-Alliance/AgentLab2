"""Storage event-file layout tests — save_event / load_event,
crash-safe save_metadata roundtrip.

The historical `Trajectory(events=[...])` round-trip tests have been
moved to `tests/test_episode_view.py`, which exercises the canonical
`TrajectoryMetadata + TrajectoryView` path. This file now only covers the
low-level `save_event` / `load_event` storage methods directly.
"""

import warnings
from pathlib import Path

import pytest
from cube.core import Action, EnvironmentOutput, Observation
from litellm import Message

from cube_harness.core import (
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
    TrajectoryMetadata,
)
from cube_harness.llm import LLMCall, LLMConfig, Prompt, Usage
from cube_harness.storage import EVENTS_DIR, FileStorage


def _agent_event(tag: str = "act") -> LLMCallEvent:
    call = LLMCall(
        tag=tag,
        llm_config=LLMConfig(model_name="openai/gpt-4o-mini"),
        prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
        output=Message(content="ok", role="assistant"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost=0.0),
    )
    _ = Action  # silence unused
    _ = EnvironmentOutput
    return LLMCallEvent(call=call)


def _tool_call_event(parent_id: str) -> ToolCallEvent:
    return ToolCallEvent(
        parent_event_id=parent_id,
        action_id="a-1",
        obs=Observation(),
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

    parent = _agent_event(tag="my_tag")
    te = TrajectoryEvent(output=parent, start_time=0.0, end_time=0.1)
    storage.save_event(te, "t")

    # Loaded event round-trips through msgpack+zstd serialization.
    loaded = storage.load_event("t", 0)
    assert isinstance(loaded.output, LLMCallEvent)
    assert loaded.output.call is not None
    assert loaded.output.call.tag == "my_tag"


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
    assert "000_llm.msgpack.zst" in names
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


def test_serialize_event_emits_no_spurious_union_warnings() -> None:
    """Regression: persisting a TrajectoryEvent must not flood stdout.

    `TrajectoryEvent.output` is a polymorphic `TypedBaseModel` union
    (LLMCall/Tool/Eval/AgentError), with further nested unions inside
    `LLMCall`/`Message`. Without `serialize_as_any=True`, pydantic's
    smart-union serializer trials every member and emits a
    `PydanticSerializationUnexpectedValue` warning per non-match —
    ~24 warning lines for a single LLMCallEvent, on *every* event persist.
    The on-disk payload is unchanged (`_type` still round-trips); only
    the noise must be gone.
    """
    from cube_harness.storage import _serialize_event

    ev = TrajectoryEvent(output=_agent_event(), start_time=0.0, end_time=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _serialize_event(ev)
    spurious = [w for w in caught if "PydanticSerializationUnexpectedValue" in str(w.message)]
    assert not spurious, f"event serialization emitted {len(spurious)} spurious union warning(s)"
