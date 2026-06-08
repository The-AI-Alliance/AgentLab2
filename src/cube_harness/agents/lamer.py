"""LaMer-style agents: cross-episode self-reflection for multi-episode rollouts.

A LaMer agent, at the end of a failed episode, reflects on its own attempt (``finalize`` hook, per
``openspec/changes/multi-episode-rollouts/``), persists the lesson to a file, and injects accumulated
lessons into the next episode. The reflection *mechanism* is the shared
:class:`~cube_harness.agents.reflection.CrossEpisodeReflector`; the domain *content* (verdict, prompt
wording, how a failed attempt is read from history) comes from a
:class:`~cube_harness.agents.reflection.ReflectionProvider`, injected via ``reflection_provider`` on
the config. Two variants ship here — :class:`LaMerTirAgent` (tool-calling, injects lessons into the
system prompt) and :class:`LaMerReactAgent` (ReAct, splices lessons in as messages) — both pure
mechanism, fully domain-agnostic.

Designed for RL training (LaMer): the reflection LLM call is made through the agent's training LLM so
it carries token-ids + logprobs, and is returned as an ``AgentOutput`` so ``Episode`` records it as a
trajectory step — which the PipelineRL ``cube_rl`` rollout turns into a trainable example. The
multi-episode loop + per-rollout ``memory_path`` are the recipe/rollout's job (see the RFC: sequential
N-episode runs are recipe-level for-loops over ``Episode.run()``, via :func:`run_multi_episode_rollout`).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cube.core import ActionSchema
from litellm import Message

from cube_harness.agents.react import ReactAgent, ReactAgentConfig
from cube_harness.agents.reflection import CrossEpisodeReflector, ReflectionProviderConfig
from cube_harness.agents.tir import TirAgent, TirAgentConfig
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.episode import MAX_STEPS, Episode

logger = logging.getLogger(__name__)


def _reflection_output(call: Any) -> AgentOutput | None:
    """Wrap a reflection ``LLMCall`` (or None) as an ``AgentOutput`` for the trajectory."""
    if call is None:
        return None
    return AgentOutput(actions=[], llm_calls=[call], thoughts=(getattr(call.output, "content", None) or "").strip())


class LaMerTirAgentConfig(TirAgentConfig):
    """TirAgent + cross-episode self-reflection. ``reflection_provider`` (a serializable
    :class:`ReflectionProviderConfig`) supplies the domain reflection content; ``memory_path`` is the
    JSON file the agent reads on construction and appends to on ``finalize`` — the recipe/rollout
    points it at a fresh per-rollout path so the N episodes of one rollout share lessons.
    """

    reflection_provider: ReflectionProviderConfig
    memory_path: Path | None = None

    @property
    def agent_name(self) -> str:
        variant = getattr(self.reflection_provider, "reflection_variant", None)
        return f"LaMerTir-{variant}" if variant else "LaMerTir"

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: object) -> "LaMerTirAgent":
        _ = kwargs
        return LaMerTirAgent(config=self, tools=action_set or [])


class LaMerTirAgent(TirAgent):
    name: str = "lamer_tir_agent"
    description: str = "TIR agent with cross-episode self-reflection via finalize()."

    def __init__(self, config: LaMerTirAgentConfig, tools: list[ActionSchema]) -> None:
        super().__init__(config, tools)
        self._lamer_config = config
        self._reflector = CrossEpisodeReflector(config.reflection_provider.make(), config.memory_path)

    @property
    def memory(self) -> list[str]:
        return self._reflector.memory

    def _build_prompt_messages(self) -> list[dict | Message]:
        """Inject accumulated reflections into the system prompt (else the base TIR prompt)."""
        system = self.config.system_prompt
        lessons = self._reflector.lessons(max_actions=self.config.max_actions)
        if lessons:
            system = f"{system}\n\n{lessons}" if system else lessons
        messages: list[dict | Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self.history)
        return messages

    def finalize(self, reward: float) -> AgentOutput | None:
        """On failure, reflect (a trainable LLM call) and append the lesson to memory.

        Success (reward > 0) ⇒ no reflection. Returns the reflection as an ``AgentOutput`` so
        ``Episode`` records it; the next episode in the rollout reads ``memory_path`` and injects it.
        """
        call = self._reflector.reflect(
            llm=self.llm,
            llm_config=self.config.llm_config,
            history=self.history,
            reward=reward,
            max_actions=self.config.max_actions,
            requirements=self.config.system_prompt,
        )
        return _reflection_output(call)


class LaMerReactAgentConfig(ReactAgentConfig):
    """ReactAgent + cross-episode self-reflection. Same ``reflection_provider`` / ``memory_path``
    contract as :class:`LaMerTirAgentConfig`; lessons are spliced into the rendered history as
    messages (the ReAct idiom) rather than appended to the system prompt."""

    reflection_provider: ReflectionProviderConfig
    memory_path: Path | None = None

    @property
    def agent_name(self) -> str:
        return f"LaMer-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: object) -> "LaMerReactAgent":
        _ = kwargs
        return LaMerReactAgent(config=self, tools=action_set or [])


class LaMerReactAgent(ReactAgent):
    name: str = "lamer_react_agent"
    description: str = "ReAct agent with cross-episode self-reflection via finalize()."

    def __init__(self, config: LaMerReactAgentConfig, tools: list[ActionSchema]) -> None:
        super().__init__(config, tools)
        self._lamer_config = config
        self._reflector = CrossEpisodeReflector(config.reflection_provider.make(), config.memory_path)

    @property
    def memory(self) -> list[str]:
        return self._reflector.memory

    def finalize(self, reward: float) -> AgentOutput | None:
        call = self._reflector.reflect(
            llm=self.llm,
            llm_config=self.config.llm_config,
            history=self.history,
            reward=reward,
            max_actions=self.config.max_actions,
            requirements=self.config.system_prompt,
        )
        return _reflection_output(call)

    def choose_steps_to_render(self, history: list[dict | Message]) -> list[dict | Message]:
        """Splice accumulated lessons in as a user message right after the system prompt."""
        base = super().choose_steps_to_render(history)
        lessons = self._reflector.lessons(max_actions=self.config.max_actions)
        if not lessons or not base:
            return base
        return [
            base[0],  # system message
            {"role": "user", "content": lessons},
            {"role": "assistant", "content": "I'll keep these lessons in mind."},
            *base[1:],
        ]


def _episode_solved(trajectory: Trajectory) -> bool:
    """Default early-stop: the episode's cube reward is positive."""
    return float((getattr(trajectory, "reward_info", None) or {}).get("reward", 0.0)) > 0.0


def run_multi_episode_rollout(
    agent_config: Any,
    task_config: Any,
    runtime_context: Any,
    k_episodes: int,
    *,
    max_steps: int = MAX_STEPS,
    container_backend: Any = None,
    output_dir: str | Path = "",
    exp_name: str = "rollout",
    persist_episode: bool = False,
    storage: Any = None,
    memory_path: Path | None = None,
    early_stop: Callable[[Trajectory], bool] = _episode_solved,
) -> list[Trajectory]:
    """Run up to ``k_episodes`` sequential ``Episode.run()`` calls against the SAME ``task_config``,
    sharing one cross-episode memory file, and stop early once ``early_stop(trajectory)`` is True.

    This is the recipe-level multi-episode loop the RFC mandates (``Episode`` stays single-episode):
    each iteration is a fresh ``Episode`` (fresh agent + ``task.reset()``); cross-episode state
    survives ONLY through the shared memory file, which a fresh LaMer agent re-reads each episode.

    Memory handling: if the agent supports ``memory_path`` and the caller didn't supply one (and
    ``k>1``), a throwaway temp file is created and cleaned up. A caller-supplied ``memory_path`` (or
    one already set on ``agent_config``) is respected and left in place (e.g. a persisted recipe path).

    With ``k_episodes==1`` — or an ``agent_config`` without ``memory_path`` (a plain ``TirAgent``) —
    this degenerates to a single ordinary rollout, so non-LaMer agents work unchanged.

    Returns the per-episode trajectories in order (length 1..k_episodes).
    """
    k = max(1, int(k_episodes))
    supports_memory = hasattr(agent_config, "memory_path")
    if memory_path is not None and supports_memory:
        agent_config = agent_config.model_copy(update={"memory_path": memory_path})

    tmp_memory_dir: str | None = None
    if k > 1 and supports_memory and getattr(agent_config, "memory_path", None) is None:
        tmp_memory_dir = tempfile.mkdtemp(prefix="lamer_mem_")
        agent_config = agent_config.model_copy(update={"memory_path": Path(tmp_memory_dir) / "memory.json"})

    trajectories: list[Trajectory] = []
    try:
        for episode_index in range(k):
            trajectory = Episode(
                id=episode_index,
                output_dir=output_dir,
                agent_config=agent_config,
                task_config=task_config,
                exp_name=exp_name,
                max_steps=max_steps,
                storage=storage,
                runtime_context=runtime_context,
                persist_episode=persist_episode,
                container_backend=container_backend,
            ).run()
            trajectories.append(trajectory)
            if early_stop(trajectory):
                break
    finally:
        if tmp_memory_dir is not None:
            shutil.rmtree(tmp_memory_dir, ignore_errors=True)
    return trajectories


def lamer_rollout_credit(
    episode_rewards: list[float],
    turns: list[tuple[int, bool]],
    *,
    gamma: float = 1.0,
) -> list[float]:
    """Per-turn RL reward for a LaMer multi-episode rollout — the meta-RL credit policy.

    Pure and framework-agnostic (an RL trainer maps its own training units onto this): ``turns`` is
    one ``(episode_index, is_reflection)`` per training unit, ``episode_rewards`` is the per-episode
    outcome (index = episode). Returns one reward per turn, in order. Every turn is credited by the
    rollout's EVENTUAL outcome ``max(episode_rewards)`` (so a wrong-but-recoverable attempt still beats
    a completely-wrong one, which scores 0), split by turn type:

    - a REFLECTION turn earns the full eventual outcome — a reflection that led to success scores it in
      full (the meta-RL signal);
    - a SOLVE turn earns the eventual outcome discounted by distance from the terminal episode,
      ``outcome * gamma**(n - 1 - e)``: the successful (last) attempt gets it in full, an earlier
      wrong-but-recoverable attempt gets a ``gamma``-discounted positive, a never-recovered attempt
      gets 0.

    ``gamma`` (<1) discounts EARLIER solve attempts more, so a first-try success (``gamma**0``) beats a
    wrong-but-recoverable earlier attempt (``gamma**(>0)``) beats completely-wrong (0); ``gamma=1.0``
    makes recoverable == first-try (no separation).
    """
    n = len(episode_rewards)
    if n == 0:
        return [0.0] * len(turns)
    outcome = max(episode_rewards)
    return [outcome * (1.0 if is_reflection else gamma ** (n - 1 - e)) for e, is_reflection in turns]


def episodes_to_success(episode_rewards: list[float]) -> int:
    """Number of episodes (attempts) until the rollout first succeeds, or 0 if it never does.

    ``run_multi_episode_rollout`` early-stops on the first solved episode, so a successful rollout's
    last episode IS the solve and ``len(episode_rewards)`` equals the attempt count; an unsolved
    rollout returns 0. A LaMer-efficiency metric — for k=1 it degenerates to a 0/1 solve indicator.
    """
    return len(episode_rewards) if episode_rewards and max(episode_rewards) > 0 else 0
