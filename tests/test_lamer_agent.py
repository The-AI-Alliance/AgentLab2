"""Smoke tests for ``recipes/lamer_miniwob.py``'s LaMerAgent.

Covers the parts of LaMerAgent that don't need a real LLM:
- ``__init__`` loads memory from disk (or starts empty when no file).
- ``finalize`` calls the LLM, appends to memory, persists to disk, returns AgentOutput.
- ``finalize`` graceful failure: empty response, LLM exception.
- ``choose_steps_to_render`` injects memory after the system prompt.
- Memory file format round-trips: write → reload → same content.
"""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from litellm import Message

from cube_harness.core import AgentOutput
from cube_harness.llm import LLMConfig, LLMResponse, Usage

# recipes/ is not a package; import the recipe by path so we can pull
# LaMerAgent / LaMerAgentConfig out of it.
_RECIPES = Path(__file__).resolve().parent.parent / "recipes"
sys.path.insert(0, str(_RECIPES))
import lamer_miniwob  # noqa: E402

LaMerAgent = lamer_miniwob.LaMerAgent
LaMerAgentConfig = lamer_miniwob.LaMerAgentConfig


def teardown_module(_module: object) -> None:
    """Remove the recipes/ sys.path insertion after the test module finishes."""
    if str(_RECIPES) in sys.path:
        sys.path.remove(str(_RECIPES))


def _fake_llm_response(content: str) -> LLMResponse:
    """A real ``LLMResponse`` instance for stubbing ``self.llm``.

    ``LLMCall.usage`` is a Pydantic ``Usage`` model — Pydantic V2 rejects
    duck-typed substitutes, so we build a real ``Usage``. The Message comes
    from litellm (the same class ReactAgent stores in ``self.history``).
    """
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


def _agent(memory_path: Path | None = None) -> LaMerAgent:
    cfg = LaMerAgentConfig(
        llm_config=LLMConfig(model_name="openai/gpt-4o"),
        memory_path=memory_path,
    )
    return cfg.make([])


# ---------------------------------------------------------------------------
# Inheritance + identity
# ---------------------------------------------------------------------------


class TestLaMerAgentShape:
    def test_subclasses_reactagent(self) -> None:
        agent = _agent()
        classes = {c.__name__ for c in type(agent).__mro__}
        assert {"LaMerAgent", "ReactAgent", "Agent"} <= classes

    def test_agent_name_includes_model(self) -> None:
        cfg = LaMerAgentConfig(llm_config=LLMConfig(model_name="openai/gpt-4o"))
        assert cfg.agent_name == "LaMer-openai_gpt-4o"

    def test_default_memory_is_empty_when_no_path(self) -> None:
        agent = _agent(memory_path=None)
        assert agent.memory == []


# ---------------------------------------------------------------------------
# __init__: memory load
# ---------------------------------------------------------------------------


class TestMemoryLoad:
    def test_init_starts_empty_when_file_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"  # doesn't exist
        agent = _agent(memory_path=path)
        assert agent.memory == []

    def test_init_loads_existing_memory_file(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        path.write_text(json.dumps({"memory": ["lesson 1", "lesson 2"]}))
        agent = _agent(memory_path=path)
        assert agent.memory == ["lesson 1", "lesson 2"]

    def test_init_corrupted_file_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        path.write_text("not valid json {{{")
        # Should log but not raise; memory starts empty.
        agent = _agent(memory_path=path)
        assert agent.memory == []

    def test_init_missing_memory_key_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "lamer.json"
        path.write_text(json.dumps({"other_field": "value"}))
        agent = _agent(memory_path=path)
        assert agent.memory == []


# ---------------------------------------------------------------------------
# finalize: LLM call, memory append, persistence, return value
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_returns_agent_output_with_reflection_tag(self, tmp_path: Path) -> None:
        agent = _agent(memory_path=tmp_path / "lamer.json")
        with patch.object(agent, "llm", return_value=_fake_llm_response("clicked the wrong button")):
            result = agent.finalize(reward=0.0)
        assert isinstance(result, AgentOutput)
        assert result.thoughts == "clicked the wrong button"
        assert len(result.llm_calls) == 1
        assert result.llm_calls[0].tag == "reflection"
        assert result.actions == []

    def test_finalize_appends_to_memory(self, tmp_path: Path) -> None:
        agent = _agent(memory_path=tmp_path / "lamer.json")
        assert agent.memory == []
        with patch.object(agent, "llm", return_value=_fake_llm_response("first lesson")):
            agent.finalize(reward=0.0)
        assert agent.memory == ["first lesson"]

    def test_finalize_persists_memory_to_disk(self, tmp_path: Path) -> None:
        memory_path = tmp_path / "agent_state" / "lamer.json"
        agent = _agent(memory_path=memory_path)
        with patch.object(agent, "llm", return_value=_fake_llm_response("a lesson")):
            agent.finalize(reward=0.0)
        # The file should exist with the memory persisted.
        assert memory_path.exists()
        data = json.loads(memory_path.read_text())
        assert data == {"memory": ["a lesson"]}

    def test_finalize_persists_round_trip(self, tmp_path: Path) -> None:
        """Write memory in one agent, read it back in a fresh agent — same content."""
        path = tmp_path / "lamer.json"
        agent1 = _agent(memory_path=path)
        with patch.object(agent1, "llm", return_value=_fake_llm_response("lesson A")):
            agent1.finalize(reward=0.0)
        with patch.object(agent1, "llm", return_value=_fake_llm_response("lesson B")):
            agent1.finalize(reward=1.0)
        # Fresh agent loads from disk.
        agent2 = _agent(memory_path=path)
        assert agent2.memory == ["lesson A", "lesson B"]

    def test_finalize_empty_response_does_not_append(self, tmp_path: Path) -> None:
        agent = _agent(memory_path=tmp_path / "lamer.json")
        with patch.object(agent, "llm", return_value=_fake_llm_response("   ")):
            result = agent.finalize(reward=0.0)
        assert agent.memory == []
        # Empty response → returns None (nothing to record in trajectory).
        assert result is None

    def test_finalize_llm_failure_does_not_propagate(self, tmp_path: Path) -> None:
        agent = _agent(memory_path=tmp_path / "lamer.json")
        with patch.object(agent, "llm", side_effect=RuntimeError("provider down")):
            result = agent.finalize(reward=0.0)
        # Failure → returns None; memory unchanged.
        assert result is None
        assert agent.memory == []
        # File should still be persisted (with empty memory) so subsequent runs see it.
        path = agent._lamer_config.memory_path
        assert path is not None
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"memory": []}

    def test_finalize_with_no_memory_path_does_not_crash(self) -> None:
        """When memory_path is None (e.g., the agent was constructed for ad-hoc use),
        finalize should still work — just no persistence."""
        agent = _agent(memory_path=None)
        with patch.object(agent, "llm", return_value=_fake_llm_response("lesson")):
            result = agent.finalize(reward=0.0)
        assert result is not None
        assert agent.memory == ["lesson"]

    def test_finalize_reward_in_prompt(self, tmp_path: Path) -> None:
        """The reward should be interpolated into the reflection prompt."""
        agent = _agent(memory_path=tmp_path / "lamer.json")
        captured_prompts: list[Any] = []

        def _capture(prompt):
            captured_prompts.append(prompt)
            return _fake_llm_response("reflected")

        with patch.object(agent, "llm", side_effect=_capture):
            agent.finalize(reward=0.7)
        # The user-content message in the prompt should mention reward=0.7.
        assert len(captured_prompts) == 1
        user_msgs = [m for m in captured_prompts[0].messages if m.get("role") == "user"]
        assert user_msgs
        assert "0.7" in user_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Memory injection into prompts
# ---------------------------------------------------------------------------


class TestMemoryInjection:
    def test_choose_steps_to_render_without_memory_passes_through(self) -> None:
        agent = _agent()
        agent.history = [{"role": "user", "content": "goal"}]
        msgs = agent.choose_steps_to_render(agent.history)
        # No "Lessons from previous attempts" block when memory is empty.
        assert not any(isinstance(m, dict) and "Lessons from previous attempts" in m.get("content", "") for m in msgs)

    def test_choose_steps_to_render_injects_memory_after_system(self) -> None:
        agent = _agent()
        agent.history = [{"role": "user", "content": "goal"}]
        agent.memory = ["last time I clicked the wrong button"]
        msgs = agent.choose_steps_to_render(agent.history)
        # msgs[0] = system; msgs[1] = the memory block; msgs[2] = assistant ack.
        assert isinstance(msgs[1], dict)
        assert msgs[1]["role"] == "user"
        assert "Lessons from previous attempts" in msgs[1]["content"]
        assert "last time I clicked the wrong button" in msgs[1]["content"]
        assert isinstance(msgs[2], dict)
        assert msgs[2]["role"] == "assistant"

    def test_memory_block_format_includes_attempt_numbers(self) -> None:
        agent = _agent()
        agent.history = [{"role": "user", "content": "goal"}]
        agent.memory = ["L1", "L2", "L3"]
        msgs = agent.choose_steps_to_render(agent.history)
        block = msgs[1]["content"]
        assert "Reflection after attempt 1" in block
        assert "Reflection after attempt 2" in block
        assert "Reflection after attempt 3" in block


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
