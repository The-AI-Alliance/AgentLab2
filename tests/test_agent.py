"""Tests for cube_harness.agent module."""

import json

import pytest
from cube.core import Action, Content, Observation
from PIL import Image

from cube_harness.agent import Agent, AgentConfig, apply_description_overrides
from cube_harness.core import AgentOutput


def _encoded_tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {}}}


class TestApplyDescriptionOverrides:
    def test_replaces_matching_descriptions(self) -> None:
        tools = [_encoded_tool("click", "Click."), _encoded_tool("type_text", "Type.")]
        apply_description_overrides(tools, {"click": "Click an element by its id."})
        assert tools[0]["function"]["description"] == "Click an element by its id."
        assert tools[1]["function"]["description"] == "Type."  # untouched

    def test_empty_overrides_is_noop(self) -> None:
        tools = [_encoded_tool("click", "Click.")]
        apply_description_overrides(tools, {})
        assert tools[0]["function"]["description"] == "Click."

    def test_unknown_key_raises(self) -> None:
        tools = [_encoded_tool("click", "Click.")]
        with pytest.raises(ValueError, match="unknown actions"):
            apply_description_overrides(tools, {"clik": "typo"})


class TestAgentConfig:
    """Tests for AgentConfig abstract class through MockAgentConfig."""

    def test_agent_config_make(self, mock_agent_config):
        """Test creating agent from config."""
        agent = mock_agent_config.make()
        assert agent.config == mock_agent_config

    def test_agent_config_serialization(self, mock_agent_config):
        """Test AgentConfig JSON serialization."""
        json_str = mock_agent_config.model_dump_json()
        data = json.loads(json_str)
        assert data["name"] == "mock_agent"


class TestAgent:
    """Tests for Agent abstract class through MockAgent."""

    def test_agent_initialization(self, mock_agent, mock_agent_config):
        """Test Agent initialization."""
        assert mock_agent.config == mock_agent_config
        assert mock_agent.step_count == 0

    def test_agent_class_attributes(self, mock_agent):
        """Test Agent class attributes."""
        assert mock_agent.name == "MockAgent"
        assert mock_agent.description == "A mock agent for testing"
        assert "text" in mock_agent.input_content_types
        assert "action" in mock_agent.output_content_types

    def test_agent_step(self, mock_agent):
        """Test Agent step method."""
        obs = Observation.from_text("Click the button")
        output = mock_agent.step(obs)

        assert isinstance(output, AgentOutput)
        assert mock_agent.step_count == 1

    def test_agent_step_multiple(self, mock_agent):
        """Test multiple Agent step calls."""
        for i in range(5):
            obs = Observation.from_text(f"Step {i}")
            mock_agent.step(obs)

        assert mock_agent.step_count == 5

    def test_agent_step_returns_configured_actions(self, mock_agent):
        """Test that agent returns configured actions."""
        mock_agent.actions_to_return = [
            Action(name="click", arguments={"element_id": "btn1"}),
            Action(name="type_text", arguments={"text": "hello"}),
        ]

        obs = Observation.from_text("Do something")
        output = mock_agent.step(obs)

        assert len(output.actions) == 2
        assert output.actions[0].name == "click"
        assert output.actions[1].name == "type_text"

    def test_agent_step_default_stop_action(self, mock_agent):
        """Test that agent returns stop action by default."""
        obs = Observation.from_text("Do something")
        output = mock_agent.step(obs)

        assert len(output.actions) == 1
        assert output.actions[0].name == "final_step"

    def test_agent_repr(self, mock_agent):
        """Test Agent string representation."""
        repr_str = repr(mock_agent)
        # Should be JSON from config
        assert "mock_agent" in repr_str

    def test_agent_with_image_observation(self, mock_agent):
        """Test Agent handling observation with image."""
        img = Image.new("RGB", (100, 100), color="red")
        obs = Observation(contents=[Content.from_data("Click the red area"), Content.from_data(img, name="screenshot")])

        output = mock_agent.step(obs)
        assert isinstance(output, AgentOutput)

    def test_agent_config_inheritance(self):
        """Test that AgentConfig can be extended with additional fields."""

        class ExtendedConfig(AgentConfig):
            custom_param: str = "default"
            another_param: int = 42

            def make(self) -> "ExtendedAgent":
                return ExtendedAgent(config=self)

        class ExtendedAgent(Agent):
            name = "ExtendedAgent"
            description = "Extended test agent"
            input_content_types = ["text"]
            output_content_types = ["action"]

            def step(self, obs: Observation) -> AgentOutput:
                return AgentOutput(actions=[])

        config = ExtendedConfig(custom_param="custom_value", another_param=100)
        agent = config.make()

        assert agent.config.custom_param == "custom_value"
        assert agent.config.another_param == 100

    def test_agent_step_with_tool_call_results(self, mock_agent):
        """Test Agent step with observation containing tool call results."""
        obs = Observation(
            contents=[
                Content.from_data("Initial instruction"),
                Content.from_data("Tool result 1", tool_call_id="call_1"),
                Content.from_data("Tool result 2", tool_call_id="call_2"),
            ]
        )

        output = mock_agent.step(obs)
        assert isinstance(output, AgentOutput)


class TestAgentFinalize:
    """Tests for the Agent.finalize() end-of-episode hook.

    Spec: ``openspec/changes/multi-episode-rollouts/`` — additive default-no-op
    method on the base Agent ABC. Default returns None. Subclasses may override
    to return an AgentOutput; Episode appends non-None returns as synthetic
    trajectory steps.
    """

    def test_default_finalize_returns_none(self, mock_agent) -> None:
        """Default Agent.finalize() implementation returns None — backwards compatible."""
        result = mock_agent.finalize(reward=1.0)
        assert result is None

    def test_default_finalize_with_zero_reward(self, mock_agent) -> None:
        """Default no-op shape doesn't depend on reward sign."""
        assert mock_agent.finalize(reward=0.0) is None
        assert mock_agent.finalize(reward=-0.5) is None
        assert mock_agent.finalize(reward=1.5) is None

    def test_subclass_finalize_can_return_agent_output(self) -> None:
        """A subclass that overrides finalize can return an AgentOutput."""
        from litellm import Message

        from cube_harness.llm import LLMCall, LLMConfig, Prompt, Usage

        class _FinalizingAgent(Agent):
            name = "finalizing"
            description = "test"
            input_content_types = ["text"]
            output_content_types = ["action"]

            def step(self, obs: Observation) -> AgentOutput:
                _ = obs
                return AgentOutput()

            def finalize(self, reward: float) -> AgentOutput | None:
                # Construct a real LLMCall so this also exercises the tag plumbing
                # that XRay and cost stats rely on.
                call = LLMCall(
                    tag="reflection",
                    llm_config=LLMConfig(model_name="openai/gpt-4o"),
                    prompt=Prompt(messages=[{"role": "user", "content": f"reward={reward}"}]),
                    output=Message(role="assistant", content=f"got {reward}"),
                    usage=Usage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        cached_tokens=0,
                        cache_creation_tokens=0,
                        cost=0.001,
                    ),
                )
                return AgentOutput(actions=[], llm_calls=[call], thoughts=f"reward={reward}")

        class _DummyConfig(AgentConfig):
            def make(self, action_set=None, **kwargs):
                _ = action_set, kwargs
                return _FinalizingAgent(config=self)

        agent = _DummyConfig().make()
        result = agent.finalize(reward=1.0)
        assert isinstance(result, AgentOutput)
        assert result.actions == []
        assert len(result.llm_calls) == 1
        assert result.llm_calls[0].tag == "reflection"
        assert result.thoughts == "reward=1.0"

    def test_subclass_finalize_can_still_return_none(self) -> None:
        """A subclass that overrides finalize for pure side-effects can return None."""

        class _SideEffectAgent(Agent):
            name = "se"
            description = "test"
            input_content_types = ["text"]
            output_content_types = ["action"]
            persisted_rewards: list[float] = []

            def step(self, obs: Observation) -> AgentOutput:
                _ = obs
                return AgentOutput()

            def finalize(self, reward: float) -> None:
                # Pure side-effect: record the reward, don't return an AgentOutput.
                # This is what file-based-memory agents do — flush to disk in
                # finalize(), no LLM call, return None.
                _SideEffectAgent.persisted_rewards.append(reward)
                return None

        class _Cfg(AgentConfig):
            def make(self, action_set=None, **kwargs):
                _ = action_set, kwargs
                return _SideEffectAgent(config=self)

        _SideEffectAgent.persisted_rewards = []  # reset class-level state
        agent = _Cfg().make()
        assert agent.finalize(reward=0.7) is None
        assert _SideEffectAgent.persisted_rewards == [0.7]
