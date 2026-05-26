"""Multi-episode rollout orchestrator.

A ``Rollout`` runs ``n_episodes`` episodes against the same ``Task`` with a
long-lived ``Agent`` instance. Between episodes, ``Task.reset()`` returns the
environment to its initial state and ``Agent.reflect()`` lets the agent fold
the just-finished trajectory into in-context memory used by subsequent steps.

The aggregate signal is the discounted sum ``Σ γ^k · r_k`` across episodes —
see the ``multi-episode-rollouts`` change in ``openspec/changes/`` for the
RFC, motivation (LaMer / arxiv 2512.16848), and contract details.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from cube.benchmark import RuntimeContext
from cube.core import EnvironmentOutput, StepError, ValidatedConfig
from cube.task import TaskConfig
from pydantic import Field
from termcolor import colored

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput, Trajectory, TrajectoryStep
from cube_harness.metrics.tracer import get_tracer
from cube_harness.storage import FileStorage

logger = logging.getLogger(__name__)

# Match Episode's default upper bound; overridable per-rollout via RolloutConfig.
DEFAULT_MAX_STEPS_PER_EPISODE = 100


class RolloutConfig(ValidatedConfig):
    """Configuration for a Rollout.

    ``gamma`` is the cross-episode discount used to aggregate per-episode
    rewards into ``RolloutResult.discounted_reward``. ``gamma=1.0`` reduces
    to a plain sum.
    """

    n_episodes: int = Field(ge=1)
    gamma: float = Field(default=1.0, gt=0.0, le=1.0)
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE


class RolloutResult(ValidatedConfig):
    """Result returned by ``Rollout.run()``.

    ``trajectories`` are emitted in execution order; the k-th entry was
    produced under whatever agent state existed after ``k-1`` reflections.
    """

    rollout_id: str
    trajectories: list[Trajectory]
    per_episode_rewards: list[float]
    discounted_reward: float
    agent_config_dump: dict[str, Any] = Field(default_factory=dict)


class Rollout:
    """N-episode rollout with a persistent ``Agent``.

    The same ``Agent`` instance handles every episode; ``Task.reset()`` is
    invoked before each episode and ``Agent.reflect()`` after each non-final
    episode. The task is constructed once and closed once; the agent never
    crosses a process boundary inside ``run()``.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        agent_config: AgentConfig,
        config: RolloutConfig,
        output_dir: Path | None = None,
        runtime_context: RuntimeContext | None = None,
        exp_name: str = "rollout",
    ) -> None:
        self.task_config = task_config
        self.agent_config = agent_config
        self.config = config
        self.output_dir = output_dir
        self.runtime_context = runtime_context
        self.exp_name = exp_name

    def run(self) -> RolloutResult:
        """Execute ``n_episodes`` episodes with a persistent agent."""
        rollout_id = str(uuid.uuid4())
        logger.info(
            colored(
                f"Rollout {rollout_id[:8]} start — task={self.task_config.task_id} "
                f"n_episodes={self.config.n_episodes} gamma={self.config.gamma}",
                "blue",
            )
        )

        # Task and agent are constructed ONCE and reused across all N episodes.
        task = self.task_config.make(runtime_context=self.runtime_context)
        agent = self.agent_config.make(
            action_set=task.action_set,
            task_id=self.task_config.task_id,
        )

        trajectories: list[Trajectory] = []
        rewards: list[float] = []
        try:
            for episode_idx in range(self.config.n_episodes):
                logger.info(colored(f"  Episode {episode_idx + 1}/{self.config.n_episodes}", "cyan"))
                trajectory, reward = self._run_episode(
                    task=task, agent=agent, rollout_id=rollout_id, episode_idx=episode_idx
                )
                trajectories.append(trajectory)
                rewards.append(reward)
                # reflect() is called between episodes only — never after the final one.
                if episode_idx < self.config.n_episodes - 1:
                    refl_ts = time.time()
                    reflection_output = agent.reflect(trajectory, reward)
                    if reflection_output is not None:
                        # Record the reflection as a synthetic agent step on the
                        # just-finished trajectory so its LLMCall is observable in
                        # XRay, cost stats, and training-data extraction.
                        trajectory.steps.append(
                            TrajectoryStep(
                                output=reflection_output,
                                start_time=refl_ts,
                                end_time=time.time(),
                            )
                        )
        finally:
            task.close()

        discounted = sum((self.config.gamma**k) * r for k, r in enumerate(rewards))
        result = RolloutResult(
            rollout_id=rollout_id,
            trajectories=trajectories,
            per_episode_rewards=rewards,
            discounted_reward=discounted,
            agent_config_dump=self.agent_config.model_dump(mode="json", serialize_as_any=True),
        )
        logger.info(
            colored(
                f"Rollout {rollout_id[:8]} done — rewards={rewards} discounted={discounted:.4f}",
                "blue",
            )
        )

        if self.output_dir is not None:
            self._persist(result)
        return result

    def _run_episode(
        self,
        task: Any,
        agent: Agent,
        rollout_id: str,
        episode_idx: int,
    ) -> tuple[Trajectory, float]:
        """Run one episode against the persistent ``task`` and ``agent``.

        Mirrors ``Episode._run_loop``'s core loop: agent.step → task.step until
        ``done`` or the per-episode step cap, with a final ``evaluate()`` if the
        cap fires before the env signals done so the trajectory carries the
        true reward.
        """
        trajectory_id = f"{rollout_id[:8]}_ep{episode_idx}"
        tracer = get_tracer(self.exp_name)

        with tracer.episode(self.task_config.task_id, experiment=self.exp_name):
            start_time = time.time()
            obs, info = task.reset()
            env_output = EnvironmentOutput(obs=obs, info=info)

            trajectory = Trajectory(
                id=trajectory_id,
                steps=[TrajectoryStep(output=env_output, start_time=start_time, end_time=time.time())],
                metadata={
                    "task_id": self.task_config.task_id,
                    "agent_name": self.agent_config.agent_name,
                    "rollout_id": rollout_id,
                    "episode_index_in_rollout": episode_idx,
                    **env_output.info,
                },
                start_time=start_time,
            )

            turns = 0
            while not env_output.done and turns < self.config.max_steps_per_episode:
                ts = time.time()
                try:
                    agent_output = agent.step(env_output.obs)
                except Exception as e:
                    logger.exception(f"agent.step() error at turn {turns}")
                    agent_output = AgentOutput(error=StepError.from_exception(e))
                    trajectory.steps.append(TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time()))
                    break

                trajectory.steps.append(TrajectoryStep(output=agent_output, start_time=ts, end_time=time.time()))
                if agent_output.error is not None:
                    break
                if not agent_output.actions:
                    logger.info(colored("Agent returned no actions — stopping episode.", "yellow"))
                    break

                env_ts = time.time()
                try:
                    env_output = task.step(agent_output.actions)
                except Exception as e:
                    logger.exception(f"task.step() error at turn {turns}")
                    env_output = EnvironmentOutput(obs=env_output.obs, error=StepError.from_exception(e))
                    trajectory.steps.append(TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time()))
                    break

                trajectory.steps.append(TrajectoryStep(output=env_output, start_time=env_ts, end_time=time.time()))
                turns += 1

            # If the cap fired before done=True, force a final evaluate() so the
            # trajectory's last EnvironmentOutput carries the real reward.
            if not env_output.done:
                try:
                    eval_ts = time.time()
                    forced_reward, forced_info = task.evaluate(env_output.obs)
                    env_output = EnvironmentOutput(
                        obs=env_output.obs,
                        reward=forced_reward,
                        done=env_output.done,
                        info={**env_output.info, **forced_info},
                        error=env_output.error,
                    )
                    trajectory.steps.append(TrajectoryStep(output=env_output, start_time=eval_ts, end_time=time.time()))
                except Exception:
                    logger.exception("Final task.evaluate() raised; keeping prior reward")

            trajectory.end_time = time.time()
            trajectory.reward_info = {
                "reward": env_output.reward,
                "done": env_output.done,
                **env_output.info,
            }
            return trajectory, env_output.reward

    def _persist(self, result: RolloutResult) -> None:
        """Write trajectories using FileStorage so XRay discovers them, plus a
        rollout-level aggregate record that links the per-episode trajectories.

        Layout:
            <output_dir>/episodes/<trajectory_id>/            ← XRay-readable per-episode
                metadata.json + steps/NNN_*.json
            <output_dir>/rollouts/<rollout_id>/rollout_record.json   ← rollout-aware aggregate
        """
        assert self.output_dir is not None
        # 1. Per-episode trajectories in the canonical FileStorage V2 layout.
        storage = FileStorage(self.output_dir)
        for traj in result.trajectories:
            storage.save_trajectory(traj, allow_overwrite=True)
        # 2. Rollout-level aggregate so multi-episode info isn't lost.
        rollout_dir = self.output_dir / "rollouts" / result.rollout_id
        rollout_dir.mkdir(parents=True, exist_ok=True)
        record = result.model_dump(exclude={"trajectories"}, mode="json")
        record["trajectory_ids"] = [t.id for t in result.trajectories]
        record["n_episodes"] = self.config.n_episodes
        record["gamma"] = self.config.gamma
        (rollout_dir / "rollout_record.json").write_text(json.dumps(record, indent=2))
