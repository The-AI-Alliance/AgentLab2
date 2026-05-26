"""Smoke tests for the LaMer recipe's agent (recipes/lamer_miniwob.py).

Validates the parts of LaMerAgent that don't require live LLM calls:
- the ReactAgent inheritance and the agent_name override
- memory injection in choose_steps_to_render
- _render_trajectory_for_reflection produces a transcript
- the ReAct history reset that reflect() performs
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the recipe by path — recipes/ isn't a package.
_RECIPES = Path(__file__).resolve().parent.parent / "recipes"
sys.path.insert(0, str(_RECIPES))
import lamer_miniwob  # noqa: E402
from cube.core import Action  # noqa: E402
from litellm import Message  # noqa: E402

from cube_harness.agent import Agent  # noqa: E402
from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep  # noqa: E402
from cube_harness.llm import LLMConfig, LLMResponse, Usage  # noqa: E402

LaMerAgent = lamer_miniwob.LaMerAgent
LaMerAgentConfig = lamer_miniwob.LaMerAgentConfig
_render = lamer_miniwob._render_trajectory_for_reflection


def _agent() -> LaMerAgent:
    cfg = LaMerAgentConfig(llm_config=LLMConfig(model_name="gpt-4o-mini"))
    return cfg.make([])


def test_lamer_agent_subclasses_reactagent() -> None:
    """Inheritance chain is LaMer → React → Agent — required so step() works."""
    agent = _agent()
    classes = {c.__name__ for c in type(agent).__mro__}
    assert {"LaMerAgent", "ReactAgent", "Agent"} <= classes


def test_lamer_agent_overrides_reflect() -> None:
    """``reflect`` is overridden — not the base no-op."""
    assert LaMerAgent.reflect is not Agent.reflect


def test_lamer_agent_name() -> None:
    """agent_name reflects the model and is filesystem-safe."""
    cfg = LaMerAgentConfig(llm_config=LLMConfig(model_name="openai/gpt-4o"))
    assert cfg.agent_name == "LaMer-openai_gpt-4o"


def test_choose_steps_to_render_without_memory_passes_through() -> None:
    """With empty memory, the prompt is unchanged from ReactAgent's."""
    agent = _agent()
    agent.history = [{"role": "user", "content": "goal"}]
    base = agent.choose_steps_to_render(agent.history)
    # ReactAgent: system + goal + (last N steps) + react instruction
    assert any(m.get("content", "") == "goal" for m in base if isinstance(m, dict))
    # No memory block injected
    assert not any("Lessons from previous attempts" in m.get("content", "") for m in base if isinstance(m, dict))


def test_choose_steps_to_render_with_memory_injects_block() -> None:
    """A memory entry surfaces as a user message right after the system prompt."""
    agent = _agent()
    agent.history = [{"role": "user", "content": "goal"}]
    agent.memory = ["1. Don't click X. 2. Type Y first."]
    msgs = agent.choose_steps_to_render(agent.history)
    # msgs[0] = system; msgs[1] should be the memory block
    assert isinstance(msgs[1], dict)
    assert "Lessons from previous attempts" in msgs[1]["content"]
    assert "Don't click X" in msgs[1]["content"]
    # msgs[2] is the assistant ack
    assert isinstance(msgs[2], dict)
    assert msgs[2]["role"] == "assistant"


def test_render_trajectory_includes_actions_and_env_summaries() -> None:
    """Renderer emits one line per agent/env step, summarising actions and rewards."""
    traj = Trajectory(
        id="t",
        steps=[
            TrajectoryStep(
                output=AgentOutput(
                    actions=[Action(id="a", name="click", arguments={"id": "btn"})],
                    thoughts="I should click the button",
                )
            ),
        ],
        metadata={},
    )
    rendered = _render(traj)
    assert "[thoughts]" in rendered
    assert "[action]" in rendered
    assert "click" in rendered


def _fake_llm_response(content: str) -> LLMResponse:
    """A real ``LLMResponse`` for stubbing ``self.llm`` in reflect() tests.

    ``LLMCall.usage`` is typed as ``Usage`` — Pydantic V2 rejects duck-typed
    substitutes, so we build a real ``Usage`` instance. ``Message`` comes from
    litellm (same class ReactAgent stores in ``self.history``).
    """
    return LLMResponse(
        message=Message(role="assistant", content=content),
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cached_tokens=0,
            cache_creation_tokens=0,
            cost=0.001,
        ),
    )


def test_reflect_appends_memory_and_resets_history() -> None:
    """A successful reflection call appends to memory and clears per-episode ReAct state."""
    agent = _agent()
    agent.history = [{"role": "user", "content": "old"}]
    agent._actions_cnt = 5
    traj = Trajectory(id="t", steps=[], metadata={})

    with patch.object(agent, "llm", return_value=_fake_llm_response("Lesson: be patient.")):
        result = agent.reflect(traj, final_reward=0.0)

    assert agent.memory == ["Lesson: be patient."]
    assert agent.history == []
    assert agent._actions_cnt == 0
    # The reflection AgentOutput is returned so Rollout can append it as a step.
    assert isinstance(result, AgentOutput)
    assert result.thoughts == "Lesson: be patient."
    assert result.actions == []
    assert len(result.llm_calls) == 1
    assert result.llm_calls[0].tag == "reflection"
    assert result.llm_calls[0].output.content == "Lesson: be patient."


def test_reflect_llm_failure_does_not_propagate() -> None:
    """If the reflection LLM call raises, reflect() catches, returns None, and resets history."""
    agent = _agent()
    agent.history = [{"role": "user", "content": "old"}]
    traj = Trajectory(id="t", steps=[], metadata={})

    with patch.object(agent, "llm", side_effect=RuntimeError("boom")):
        result = agent.reflect(traj, final_reward=0.0)  # must not raise

    # Failure → no memory update, history reset, None returned (no synthetic step).
    assert result is None
    assert agent.memory == []
    assert agent.history == []


def test_reflect_empty_response_does_not_append_memory() -> None:
    """An empty-string LLM response is not stored as a memory entry; AgentOutput still returned."""
    agent = _agent()
    traj = Trajectory(id="t", steps=[], metadata={})
    with patch.object(agent, "llm", return_value=_fake_llm_response("   ")):
        result = agent.reflect(traj, final_reward=0.0)
    assert agent.memory == []
    # Even with no memory text, the call happened — return it so the LLMCall is recorded.
    assert isinstance(result, AgentOutput)
    assert result.thoughts is None
    assert result.llm_calls[0].tag == "reflection"


# Clean up the sys.path hack we used to import the recipe.
def teardown_module(_module: object) -> None:
    if str(_RECIPES) in sys.path:
        sys.path.remove(str(_RECIPES))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
