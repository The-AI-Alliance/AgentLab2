"""Tests for the LaMer agents (``cube_harness.agents.lamer``).

Covers the parts that don't need a real LLM, for both the ReAct and TIR variants:
- ``__init__`` loads memory from disk (or starts empty when no file).
- ``finalize`` reflects only on failure (reward <= 0): LLM call → append to memory → persist →
  return AgentOutput; success / empty response / LLM exception all return None gracefully.
- Lesson injection: ReAct splices a memory block in as messages; TIR appends to the system prompt.

The reflection *content* here comes from the generic ``SimpleReflectionProvider`` so these tests have
no cube dependency; the math ``MathReflectionProvider`` is covered in ``test_reflection.py``.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from litellm import Message

from cube_harness.agents.lamer import (
    LaMerReactAgent,
    LaMerReactAgentConfig,
    LaMerTirAgent,
    LaMerTirAgentConfig,
    episodes_to_success,
    lamer_rollout_credit,
)
from cube_harness.agents.reflection import SimpleReflectionProviderConfig
from cube_harness.core import AgentOutput
from cube_harness.llm import LLMConfig, LLMResponse, Usage


def _fake_llm_response(content: str) -> LLMResponse:
    """A real ``LLMResponse`` instance for stubbing ``self.llm`` (Pydantic rejects duck types)."""
    return LLMResponse(
        message=Message(role="assistant", content=content),
        usage=Usage(
            prompt_tokens=42,
            completion_tokens=12,
            total_tokens=54,
            cached_tokens=0,
            cache_creation_tokens=0,
            cost=0.0004,
        ),
    )


def _react_agent(memory_path: Path | None = None) -> LaMerReactAgent:
    cfg = LaMerReactAgentConfig(
        llm_config=LLMConfig(model_name="openai/gpt-4o"),
        reflection_provider=SimpleReflectionProviderConfig(),
        memory_path=memory_path,
    )
    return cfg.make([])


def _tir_agent(memory_path: Path | None = None, system_prompt: str = "BASE PROMPT") -> LaMerTirAgent:
    cfg = LaMerTirAgentConfig(
        llm_config=LLMConfig(model_name="openai/gpt-4o"),
        system_prompt=system_prompt,
        max_actions=3,
        reflection_provider=SimpleReflectionProviderConfig(),
        memory_path=memory_path,
    )
    return cfg.make([])


# ---------------------------------------------------------------------------
# Shape + identity
# ---------------------------------------------------------------------------


class TestShape:
    def test_react_subclasses_reactagent(self) -> None:
        classes = {c.__name__ for c in type(_react_agent()).__mro__}
        assert {"LaMerReactAgent", "ReactAgent", "Agent"} <= classes

    def test_tir_subclasses_tiragent(self) -> None:
        classes = {c.__name__ for c in type(_tir_agent()).__mro__}
        assert {"LaMerTirAgent", "TirAgent", "Agent"} <= classes

    def test_react_agent_name_includes_model(self) -> None:
        cfg = LaMerReactAgentConfig(
            llm_config=LLMConfig(model_name="openai/gpt-4o"),
            reflection_provider=SimpleReflectionProviderConfig(),
        )
        assert cfg.agent_name == "LaMer-openai_gpt-4o"

    def test_default_memory_is_empty_when_no_path(self) -> None:
        assert _react_agent(memory_path=None).memory == []
        assert _tir_agent(memory_path=None).memory == []


# ---------------------------------------------------------------------------
# __init__: memory load (mechanism shared via CrossEpisodeReflector)
# ---------------------------------------------------------------------------


class TestMemoryLoad:
    def test_init_starts_empty_when_file_missing(self, tmp_path: Path) -> None:
        assert _react_agent(memory_path=tmp_path / "lamer.json").memory == []

    def test_init_loads_existing_memory_file(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        path.write_text(json.dumps({"memory": ["lesson 1", "lesson 2"]}))
        assert _react_agent(memory_path=path).memory == ["lesson 1", "lesson 2"]

    def test_init_corrupted_file_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        path.write_text("not valid json {{{")
        assert _react_agent(memory_path=path).memory == []  # logs, does not raise

    def test_init_missing_memory_key_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        path.write_text(json.dumps({"other_field": "value"}))
        assert _react_agent(memory_path=path).memory == []


# ---------------------------------------------------------------------------
# finalize: reflect only on failure; LLM call, append, persistence, return value
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_returns_agent_output_with_reflection_tag(self, tmp_path: Path) -> None:
        agent = _react_agent(memory_path=tmp_path / "lamer.json")
        with patch.object(agent, "llm", return_value=_fake_llm_response("clicked the wrong button")):
            result = agent.finalize(reward=0.0)
        assert isinstance(result, AgentOutput)
        assert result.thoughts == "clicked the wrong button"
        assert len(result.llm_calls) == 1
        assert result.llm_calls[0].tag == "reflection"
        assert result.actions == []

    def test_finalize_appends_to_memory(self, tmp_path: Path) -> None:
        agent = _react_agent(memory_path=tmp_path / "lamer.json")
        with patch.object(agent, "llm", return_value=_fake_llm_response("first lesson")):
            agent.finalize(reward=0.0)
        assert agent.memory == ["first lesson"]

    def test_finalize_persists_memory_to_disk(self, tmp_path: Path) -> None:
        memory_path = tmp_path / "agent_state" / "lamer.json"
        agent = _react_agent(memory_path=memory_path)
        with patch.object(agent, "llm", return_value=_fake_llm_response("a lesson")):
            agent.finalize(reward=0.0)
        assert json.loads(memory_path.read_text()) == {"memory": ["a lesson"]}

    def test_finalize_persists_round_trip(self, tmp_path: Path) -> None:
        """Write memory in one agent, read it back in a fresh agent — same content."""
        path = tmp_path / "lamer.json"
        agent1 = _react_agent(memory_path=path)
        for lesson in ("lesson A", "lesson B"):
            with patch.object(agent1, "llm", return_value=_fake_llm_response(lesson)):
                agent1.finalize(reward=0.0)
        assert _react_agent(memory_path=path).memory == ["lesson A", "lesson B"]

    def test_finalize_success_does_not_reflect(self, tmp_path: Path) -> None:
        """reward > 0 ⇒ no reflection (memory unchanged, returns None), but the file is persisted."""
        path = tmp_path / "lamer.json"
        agent = _react_agent(memory_path=path)
        with patch.object(agent, "llm", return_value=_fake_llm_response("should not be recorded")) as llm:
            result = agent.finalize(reward=1.0)
        assert result is None
        assert agent.memory == []
        llm.assert_not_called()
        assert json.loads(path.read_text()) == {"memory": []}

    def test_finalize_empty_response_does_not_append(self, tmp_path: Path) -> None:
        agent = _react_agent(memory_path=tmp_path / "lamer.json")
        with patch.object(agent, "llm", return_value=_fake_llm_response("   ")):
            result = agent.finalize(reward=0.0)
        assert agent.memory == []
        assert result is None

    def test_finalize_llm_failure_does_not_propagate(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        agent = _react_agent(memory_path=path)
        with patch.object(agent, "llm", side_effect=RuntimeError("provider down")):
            result = agent.finalize(reward=0.0)
        assert result is None
        assert agent.memory == []
        assert json.loads(path.read_text()) == {"memory": []}  # persisted empty so later runs see it

    def test_finalize_with_no_memory_path_does_not_crash(self) -> None:
        agent = _react_agent(memory_path=None)
        with patch.object(agent, "llm", return_value=_fake_llm_response("lesson")):
            result = agent.finalize(reward=0.0)
        assert result is not None
        assert agent.memory == ["lesson"]

    def test_finalize_reward_in_prompt(self, tmp_path: Path) -> None:
        """The reward is interpolated into the SimpleReflectionProvider prompt."""
        agent = _react_agent(memory_path=tmp_path / "lamer.json")
        captured: list[Any] = []

        def _capture(prompt: Any) -> LLMResponse:
            captured.append(prompt)
            return _fake_llm_response("reflected")

        with patch.object(agent, "llm", side_effect=_capture):
            agent.finalize(reward=0.0)
        user_msgs = [m for m in captured[0].messages if m.get("role") == "user"]
        assert user_msgs and "0.0" in user_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Lesson injection — ReAct splices messages; TIR appends to the system prompt
# ---------------------------------------------------------------------------


class TestReactInjection:
    def test_choose_steps_to_render_without_memory_passes_through(self) -> None:
        agent = _react_agent()
        agent.history = [{"role": "user", "content": "goal"}]
        msgs = agent.choose_steps_to_render(agent.history)
        assert not any(isinstance(m, dict) and "Lessons from previous attempts" in m.get("content", "") for m in msgs)

    def test_choose_steps_to_render_injects_memory_after_system(self) -> None:
        agent = _react_agent()
        agent.history = [{"role": "user", "content": "goal"}]
        agent._reflector.memory = ["last time I clicked the wrong button"]
        msgs = agent.choose_steps_to_render(agent.history)
        assert isinstance(msgs[1], dict) and msgs[1]["role"] == "user"
        assert "Lessons from previous attempts" in msgs[1]["content"]
        assert "last time I clicked the wrong button" in msgs[1]["content"]
        assert isinstance(msgs[2], dict) and msgs[2]["role"] == "assistant"

    def test_memory_block_format_includes_attempt_numbers(self) -> None:
        agent = _react_agent()
        agent.history = [{"role": "user", "content": "goal"}]
        agent._reflector.memory = ["L1", "L2", "L3"]
        block = agent.choose_steps_to_render(agent.history)[1]["content"]
        assert all(f"Reflection after attempt {i}" in block for i in (1, 2, 3))


class TestTirInjection:
    def test_system_prompt_unchanged_without_memory(self) -> None:
        agent = _tir_agent(system_prompt="BASE PROMPT")
        msgs = agent._build_prompt_messages()
        assert msgs[0] == {"role": "system", "content": "BASE PROMPT"}

    def test_lessons_appended_to_system_prompt(self) -> None:
        agent = _tir_agent(system_prompt="BASE PROMPT")
        agent._reflector.memory = ["verify before submitting"]
        system = agent._build_prompt_messages()[0]["content"]
        assert system.startswith("BASE PROMPT")
        assert "Lessons from previous attempts" in system
        assert "verify before submitting" in system


# ---------------------------------------------------------------------------
# lamer_rollout_credit — the meta-RL credit policy (split + terminal-anchored discount)
# ---------------------------------------------------------------------------


class TestLamerRolloutCredit:
    def test_reflections_earn_full_eventual_outcome(self) -> None:
        # Reflections at any episode of a successful rollout get the FULL outcome (not discounted).
        assert lamer_rollout_credit([0.0, 0.0, 1.0], [(0, True), (1, True)], gamma=0.5) == [1.0, 1.0]

    def test_solve_discounted_by_distance_from_success(self) -> None:
        # Solve attempts: the successful (last) attempt full; earlier ones discounted by gamma**dist.
        turns = [(0, False), (1, False), (2, False)]  # solve0, solve1, solve2(success)
        assert lamer_rollout_credit([0.0, 0.0, 1.0], turns, gamma=0.5) == [0.25, 0.5, 1.0]

    def test_recoverable_beats_completely_wrong(self) -> None:
        # The same failed cold attempt scores gamma**dist>0 if the rollout recovers, 0 if it never does.
        recoverable = lamer_rollout_credit([0.0, 1.0], [(0, False)], gamma=0.5)[0]
        hopeless = lamer_rollout_credit([0.0, 0.0], [(0, False)], gamma=0.5)[0]
        assert recoverable == 0.5 and hopeless == 0.0 and recoverable > hopeless

    def test_first_try_beats_recoverable_when_gamma_lt_1(self) -> None:
        first_try = lamer_rollout_credit([1.0], [(0, False)], gamma=0.5)[0]  # 1 episode, solved
        recoverable = lamer_rollout_credit([0.0, 1.0], [(0, False)], gamma=0.5)[0]
        assert first_try == 1.0 and recoverable == 0.5 and first_try > recoverable

    def test_gamma_1_ties_recoverable_and_first_try(self) -> None:
        first_try = lamer_rollout_credit([1.0], [(0, False)], gamma=1.0)[0]
        recoverable = lamer_rollout_credit([0.0, 1.0], [(0, False)], gamma=1.0)[0]
        assert first_try == recoverable == 1.0  # no separation at gamma=1

    def test_all_fail_zero_credit(self) -> None:
        turns = [(0, True), (0, False), (1, True), (1, False)]
        assert lamer_rollout_credit([0.0, 0.0], turns, gamma=0.5) == [0.0, 0.0, 0.0, 0.0]

    def test_empty_episode_rewards(self) -> None:
        assert lamer_rollout_credit([], [], gamma=1.0) == []


class TestEpisodesToSuccess:
    def test_first_try_success(self) -> None:
        assert episodes_to_success([1.0]) == 1

    def test_solved_on_third_attempt(self) -> None:
        assert episodes_to_success([0.0, 0.0, 1.0]) == 3

    def test_never_solved_is_zero(self) -> None:
        assert episodes_to_success([0.0, 0.0]) == 0

    def test_empty_is_zero(self) -> None:
        assert episodes_to_success([]) == 0
