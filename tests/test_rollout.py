"""Smoke tests for cube_harness.rollout.

Validates the Rollout orchestrator's contract without touching a real cube:
- exactly ``n_episodes`` trajectories
- ``Agent.reflect()`` invoked between non-final episodes only
- agent state survives across episodes (the whole point of Rollout)
- discounted reward matches ``Σ γ^k · r_k``
- the default ``Agent.reflect()`` is a backward-compatible no-op
- ``output_dir`` persistence writes the expected files
"""

import json
from pathlib import Path
from typing import Any

import pytest
from cube.core import Action, ActionSchema, EnvironmentOutput, Observation

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.rollout import Rollout, RolloutConfig, RolloutResult

# ---------------------------------------------------------------------------
# Stubs — minimal Task/Agent that exercise the Rollout loop without a cube.
# ---------------------------------------------------------------------------


class _StubTask:
    """Toy one-shot task: reward 1.0 iff the agent submits the target value.

    ``reset()`` clears prior guesses; ``step()`` is one-shot (done=True after
    any action). The action schema declares a single ``guess(value: int)`` tool.
    """

    def __init__(self, target: int) -> None:
        self.target = target
        self.guesses: list[int] = []

    @property
    def action_set(self) -> list[ActionSchema]:
        return [
            ActionSchema(
                name="guess",
                description="Submit a guess.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            )
        ]

    def reset(self) -> tuple[Observation, dict]:
        self.guesses = []
        return Observation.from_text("Guess the secret number."), {"target": self.target}

    def step(self, actions: list[Action]) -> EnvironmentOutput:
        for a in actions:
            if a.name == "guess":
                self.guesses.append(int(a.arguments.get("value", -1)))
        hit = any(g == self.target for g in self.guesses)
        return EnvironmentOutput(
            obs=Observation.from_text(f"guesses={self.guesses}"),
            reward=1.0 if hit else 0.0,
            done=True,
            info={"guesses": list(self.guesses)},
        )

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict]:
        _ = obs
        hit = any(g == self.target for g in self.guesses)
        return (1.0 if hit else 0.0), {"guesses": list(self.guesses)}

    def close(self) -> None:
        pass


class _StubTaskConfig:
    """Minimal TaskConfig-shaped object — Rollout only needs ``.make()`` and ``.task_id``."""

    def __init__(self, target: int = 3) -> None:
        self.target = target
        self.task_id = f"stub-{target}"

    def make(self, runtime_context: Any = None) -> _StubTask:
        _ = runtime_context
        return _StubTask(target=self.target)


class _StubAgentConfig(AgentConfig):
    """Counting-stub agent config; each reflect() increments the next guess."""

    initial_guess: int = 1

    @property
    def agent_name(self) -> str:
        return "stub-agent"

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: Any) -> "_StubAgent":
        _ = kwargs
        return _StubAgent(config=self, tools=action_set or [])


class _StubAgent(Agent):
    """Guesses ``initial_guess``, then ``initial_guess+1``, ... after each reflect()."""

    name = "stub_agent"
    description = "stub"
    input_content_types = ["text/plain"]
    output_content_types = ["application/json"]

    def __init__(self, config: _StubAgentConfig, tools: list[ActionSchema]) -> None:
        super().__init__(config)
        self.tools = tools
        self.current_guess: int = config.initial_guess
        self.reflections: list[tuple[float, str]] = []

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        return AgentOutput(actions=[Action(id="g", name="guess", arguments={"value": self.current_guess})])

    def reflect(self, trajectory: Trajectory, final_reward: float) -> None:
        self.reflections.append((final_reward, f"guessed {self.current_guess}"))
        self.current_guess += 1


def _capture_constructed_agents(agent_config: _StubAgentConfig) -> list[_StubAgent]:
    """Wrap ``agent_config.make`` to record every agent it constructs.

    Returns the list that will be populated by Rollout's internal ``make`` calls.
    """
    constructed: list[_StubAgent] = []
    original_make = agent_config.make

    def capturing_make(action_set: list[ActionSchema] | None = None, **kwargs: Any) -> _StubAgent:
        agent = original_make(action_set, **kwargs)
        constructed.append(agent)
        return agent

    # Bypass Pydantic immutability by writing to __dict__ directly.
    object.__setattr__(agent_config, "make", capturing_make)
    return constructed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rollout_runs_n_episodes() -> None:
    """``Rollout.run()`` produces exactly ``n_episodes`` trajectories + rewards."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(),
        config=RolloutConfig(n_episodes=5, gamma=1.0),
    ).run()
    assert isinstance(result, RolloutResult)
    assert len(result.trajectories) == 5
    assert len(result.per_episode_rewards) == 5


def test_rollout_reflect_called_between_non_final_episodes() -> None:
    """``Agent.reflect()`` fires ``n_episodes - 1`` times — not after the final episode."""
    agent_config = _StubAgentConfig()
    constructed = _capture_constructed_agents(agent_config)
    Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=agent_config,
        config=RolloutConfig(n_episodes=4, gamma=1.0),
    ).run()
    assert len(constructed) == 1, "Agent should be constructed exactly once per rollout"
    assert len(constructed[0].reflections) == 3, (
        "reflect() should fire after episodes 0, 1, 2 — never after the final episode"
    )


def test_rollout_agent_state_persists_across_episodes() -> None:
    """The stub agent's guess increments via reflect() — proves state survives reset()."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(initial_guess=1),
        config=RolloutConfig(n_episodes=5, gamma=1.0),
    ).run()
    # ep0: guess=1 (miss), ep1: 2 (miss), ep2: 3 (hit), ep3: 4 (miss), ep4: 5 (miss).
    assert result.per_episode_rewards == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_rollout_discounted_reward_matches_formula() -> None:
    """``discounted_reward == Σ γ^k · r_k``."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(initial_guess=1),
        config=RolloutConfig(n_episodes=5, gamma=0.5),
    ).run()
    expected = sum((0.5**k) * r for k, r in enumerate(result.per_episode_rewards))
    assert result.discounted_reward == pytest.approx(expected)


def test_rollout_gamma_one_is_plain_sum() -> None:
    """``gamma=1.0`` reduces to a plain sum across episodes."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(initial_guess=2),
        config=RolloutConfig(n_episodes=3, gamma=1.0),
    ).run()
    assert result.discounted_reward == sum(result.per_episode_rewards)


def test_rollout_single_episode_skips_reflect() -> None:
    """With ``n_episodes=1``, ``reflect()`` never fires."""
    agent_config = _StubAgentConfig()
    constructed = _capture_constructed_agents(agent_config)
    Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=agent_config,
        config=RolloutConfig(n_episodes=1, gamma=0.9),
    ).run()
    assert constructed[0].reflections == []


def test_default_agent_reflect_is_noop() -> None:
    """A bare ``Agent`` (no ``reflect()`` override) is a no-op — backward compatible."""

    class _BareAgent(Agent):
        name = "bare"
        description = "bare"
        input_content_types = ["text/plain"]
        output_content_types = ["application/json"]

        def step(self, obs: Observation) -> AgentOutput:
            _ = obs
            return AgentOutput()

    bare = _BareAgent(config=_StubAgentConfig())
    # Must not raise, must return None.
    assert bare.reflect(trajectory=Trajectory(id="t"), final_reward=0.5) is None


class _ReflectingStubAgent(Agent):
    """Stub agent whose ``reflect()`` returns a synthetic ``AgentOutput`` so we can
    verify Rollout appends it as a trajectory step."""

    name = "reflecting_stub"
    description = "stub"
    input_content_types = ["text/plain"]
    output_content_types = ["application/json"]

    def __init__(self, config: _StubAgentConfig, tools: list[ActionSchema]) -> None:
        super().__init__(config)
        self.tools = tools
        self._n_reflects = 0

    def step(self, obs: Observation) -> AgentOutput:
        _ = obs
        return AgentOutput(actions=[Action(id="g", name="guess", arguments={"value": 99})])

    def reflect(self, trajectory: Trajectory, final_reward: float) -> AgentOutput | None:
        _ = trajectory, final_reward
        self._n_reflects += 1
        return AgentOutput(actions=[], llm_calls=[], thoughts=f"reflection #{self._n_reflects}")


class _ReflectingStubAgentConfig(_StubAgentConfig):
    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: Any) -> "_ReflectingStubAgent":
        _ = kwargs
        return _ReflectingStubAgent(config=self, tools=action_set or [])


def test_rollout_appends_reflection_as_synthetic_step() -> None:
    """When ``reflect()`` returns an ``AgentOutput``, Rollout appends it as a trajectory step.

    Episodes 0..N-2 should each end with a synthetic AgentOutput whose ``thoughts``
    is the reflection. The final episode has no reflection step (reflect is skipped).
    """
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_ReflectingStubAgentConfig(),
        config=RolloutConfig(n_episodes=3, gamma=1.0),
    ).run()
    # Non-final episodes carry a final reflection step; final episode does not.
    for i, traj in enumerate(result.trajectories):
        last = traj.steps[-1].output
        if i < 2:
            assert isinstance(last, AgentOutput), f"episode {i} last step should be AgentOutput"
            assert last.thoughts == f"reflection #{i + 1}"
        else:
            # Final episode's last step is the env (or forced eval) step, NOT a reflection.
            assert not isinstance(last, AgentOutput) or last.thoughts is None


def test_rollout_skips_synthetic_step_when_reflect_returns_none() -> None:
    """The default ``Agent.reflect()`` returns None — Rollout must NOT add a step."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(),  # reflect returns None implicitly (well, sets state)
        config=RolloutConfig(n_episodes=3, gamma=1.0),
    ).run()
    # With _StubAgent.reflect returning None (no `return` statement), no synthetic
    # AgentOutput should appear at the end of non-final trajectories.
    for traj in result.trajectories:
        # Last step must be the env step (done or forced eval), not a reflection AgentOutput.
        last = traj.steps[-1].output
        # If it's an AgentOutput, it must NOT be our synthetic shape (no actions + thoughts).
        if isinstance(last, AgentOutput):
            assert last.thoughts is None or last.actions, (
                "stub agent returns None from reflect — no synthetic step should be appended"
            )


def test_rollout_metadata_carries_rollout_id_and_episode_index() -> None:
    """Each trajectory carries ``rollout_id`` and ``episode_index_in_rollout`` in metadata."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(),
        config=RolloutConfig(n_episodes=3, gamma=1.0),
    ).run()
    for i, traj in enumerate(result.trajectories):
        assert traj.metadata["rollout_id"] == result.rollout_id
        assert traj.metadata["episode_index_in_rollout"] == i


def test_rollout_persists_to_output_dir(tmp_path: Path) -> None:
    """When ``output_dir`` is set, both per-episode (XRay-readable) and rollout-level
    aggregate files land on disk."""
    result = Rollout(
        task_config=_StubTaskConfig(target=3),
        agent_config=_StubAgentConfig(),
        config=RolloutConfig(n_episodes=2, gamma=0.9),
        output_dir=tmp_path,
    ).run()
    # Per-episode trajectories live in the FileStorage V2 layout XRay scans.
    for traj in result.trajectories:
        ep_dir = tmp_path / "episodes" / traj.id
        assert ep_dir.is_dir(), f"missing FileStorage episode dir for {traj.id}"
        # FileStorage V2 writes metadata + a steps/ subdir.
        assert any(p.suffix == ".json" for p in ep_dir.iterdir())
    # Rollout-level aggregate that links the trajectories.
    rollout_dir = tmp_path / "rollouts" / result.rollout_id
    assert (rollout_dir / "rollout_record.json").is_file()
    record = json.loads((rollout_dir / "rollout_record.json").read_text())
    assert record["trajectory_ids"] == [t.id for t in result.trajectories]
    assert record["n_episodes"] == 2
    assert record["gamma"] == 0.9


def test_rollout_config_rejects_invalid_values() -> None:
    """Pydantic validates ``n_episodes >= 1`` and ``0 < gamma <= 1``."""
    with pytest.raises(ValueError):
        RolloutConfig(n_episodes=0, gamma=1.0)
    with pytest.raises(ValueError):
        RolloutConfig(n_episodes=3, gamma=0.0)
    with pytest.raises(ValueError):
        RolloutConfig(n_episodes=3, gamma=1.5)
