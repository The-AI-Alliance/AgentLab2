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
    assert cards[1].title == "act"
    assert cards[2].title == "bash"

    # And the whole step groups together: select the obs -> see the LLM call.
    group = ep.group_for(2)
    assert group.llm_index == 1 and group.observation_indices == [2]
