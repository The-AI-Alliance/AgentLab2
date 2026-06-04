"""The loader recovers legacy `llm_calls` / `actions` when adapting old
V2-steps trajectories into the event stream.

Old `_act` steps carried `AgentOutput.llm_calls` + `actions`, but the shrunk
`AgentOutput` model drops those on validate. The `TrajectoryView` legacy
adapter must read them from the raw step payload and reattach them to the
synthesized `LLMCallEvent.call` / `ToolCallEvent.action` — otherwise XRay
renders empty cards for every pre-event-stream trajectory (the bug this guards).
"""

from pathlib import Path

import msgpack
import zstandard
from cube.core import Action, EnvironmentOutput, Observation
from litellm import Message

from cube_harness.analyze.xray_events import EpisodeEvents
from cube_harness.core import AgentOutput, TrajectoryMetadata
from cube_harness.llm import LLMCall, LLMConfig, Prompt, Usage
from cube_harness.storage import STEPS_DIR, FileStorage


def _write_step(path: Path, output: dict, start: float, end: float) -> None:
    step = {"output": output, "start_time": start, "end_time": end}
    path.write_bytes(zstandard.ZstdCompressor().compress(msgpack.packb(step, use_bin_type=True)))


def _legacy_llm_call() -> LLMCall:
    return LLMCall(
        tag="act",
        llm_config=LLMConfig(model_name="azure/gpt-5.4-mini"),
        prompt=Prompt(messages=[{"role": "user", "content": "list the files"}]),
        output=Message(content="I'll run ls.", role="assistant"),
        usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13, cost=0.001),
    )


def _build_legacy_episode(exp_dir: Path, traj_id: str = "legacy_task_ep0") -> None:
    storage = FileStorage(exp_dir)
    storage.save_metadata(TrajectoryMetadata(id=traj_id, metadata={"task_id": "legacy_task", "agent_name": "OldReAct"}))
    steps = storage._episode_dir(traj_id) / STEPS_DIR
    steps.mkdir(parents=True, exist_ok=True)

    # 000 reset observation
    _write_step(
        steps / "000_obs.msgpack.zst",
        EnvironmentOutput(obs=Observation.from_text("goal")).model_dump(mode="json"),
        0.0,
        0.1,
    )
    # 001 agent step — carries llm_calls + actions the new AgentOutput drops on validate
    agent = AgentOutput(actions=[Action(name="bash", arguments={"command": "ls"})]).model_dump(mode="json")
    agent["llm_calls"] = [_legacy_llm_call().model_dump(mode="json")]
    _write_step(steps / "001_act.msgpack.zst", agent, 0.1, 1.0)
    # 002 resulting observation
    _write_step(
        steps / "002_obs.msgpack.zst",
        EnvironmentOutput(obs=Observation.from_text("file1\nfile2")).model_dump(mode="json"),
        1.0,
        1.1,
    )


def test_legacy_llm_call_and_action_survive_adaptation(tmp_path: Path) -> None:
    _build_legacy_episode(tmp_path / "exp")
    view = FileStorage(tmp_path / "exp").load_episode("legacy_task_ep0")
    ep = EpisodeEvents.from_view(view)

    # [0] reset obs, [1] llm call, [2] tool-call observation
    assert len(ep) == 3
    call = ep.llm_call(1)
    assert call is not None, "legacy LLMCall was dropped — XRay would show an empty chat"
    assert call.tag == "act"
    assert call.llm_config.model_name == "azure/gpt-5.4-mini"

    action = ep.action(2)
    assert action is not None, "legacy action was dropped — observation card has no label"
    assert action.name == "bash"

    # The card titles reflect the recovered data (not the placeholder "LLM call").
    cards = ep.cards()
    assert cards[1].title == "LLM call · act"
    assert cards[2].title == "bash"

    # And the whole step groups together: select the obs -> see the LLM call.
    group = ep.group_for(2)
    assert group.llm_index == 1 and group.observation_indices == [2]


def _write_event_bytes(path: Path, payload: dict) -> None:
    path.write_bytes(zstandard.ZstdCompressor().compress(msgpack.packb(payload, use_bin_type=True)))


def test_old_agent_event_format_loads(tmp_path: Path) -> None:
    """An events/ episode using the pre-rename batched `AgentEvent` (whose
    `_type` points at a removed class) must still load: the loader synthesizes
    an LLMCallEvent from its llm_calls instead of crashing the whole view."""
    from cube_harness.core import ToolCallEvent, TrajectoryEvent
    from cube_harness.storage import EVENTS_DIR, _serialize_event

    exp = tmp_path / "exp"
    storage = FileStorage(exp)
    traj_id = "agentfmt_task_ep0"
    storage.save_metadata(TrajectoryMetadata(id=traj_id, metadata={"task_id": "t", "agent_name": "Old"}))
    events = storage._episode_dir(traj_id) / EVENTS_DIR
    events.mkdir(parents=True, exist_ok=True)

    # 000 reset observation (valid ToolCallEvent)
    reset = TrajectoryEvent(output=ToolCallEvent(parent_event_id="__reset__", obs=Observation.from_text("goal")))
    (events / "000_tool_call.msgpack.zst").write_bytes(_serialize_event(reset))
    # 001 old batched AgentEvent — _type refers to a class that no longer exists
    agent_payload = {
        "_type": "cube_harness.core.TrajectoryEvent",
        "start_time": 0.1,
        "end_time": 1.0,
        "output": {
            "_type": "cube_harness.core.AgentEvent",
            "id": "agent-1",
            "llm_calls": [_legacy_llm_call().model_dump(mode="json")],
            "actions": [Action(name="bash", arguments={"command": "ls"}).model_dump(mode="json")],
            "thoughts": "",
            "error": None,
        },
    }
    _write_event_bytes(events / "001_agent.msgpack.zst", agent_payload)
    # 002 result observation, parented on the agent event id
    obs = TrajectoryEvent(output=ToolCallEvent(parent_event_id="agent-1", obs=Observation.from_text("done")))
    (events / "002_tool_call.msgpack.zst").write_bytes(_serialize_event(obs))

    ep = EpisodeEvents.from_view(storage.load_episode(traj_id))
    assert len(ep) == 3
    call = ep.llm_call(1)
    assert call is not None and call.tag == "act"  # AgentEvent's llm_call survived
    # The result observation groups under the synthesized LLM call (shared id).
    assert ep.group_for(2).llm_index == 1
