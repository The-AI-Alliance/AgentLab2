# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "miniwob-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# miniwob-cube = { path = "../cubes/miniwob", editable = true }
# ///
"""POC recipe: LaMer-style cross-episode rollout on MiniWoB.

Runs N sequential episodes against the SAME MiniWoB task with a fresh ``LaMerReactAgent`` per
episode. Each agent reads its `memory` list from a JSON file at construction and writes back on
``finalize()`` — that file is what carries lessons across the per-rollout for-loop.

Demonstrates the `multi-episode-rollouts` design (``openspec/changes/multi-episode-rollouts/``) and
the reflection-provider protocol end-to-end:

- ``Agent.finalize(reward)`` hook runs at the end of each episode.
- Cross-episode memory + reflection mechanism: ``cube_harness.agents.reflection.CrossEpisodeReflector``.
- Domain content: the generic ``SimpleReflectionProvider`` (reward-templated reflection, plain
  lesson list) — the MiniWoB counterpart to math-tool-use's richer ``MathReflectionProvider``.
- Recipe-side nested output layout: ``<output_dir>/rollouts/<id>/{episodes,agent_state}/``.

Usage::

    .venv/bin/python recipes/lamer_miniwob.py
    .venv/bin/python recipes/lamer_miniwob.py --task-id click-button --n-episodes 3
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from miniwob_cube import MINIWOB_CONFIGS

from cube_harness import make_experiment_output_dir
from cube_harness.agents.lamer import LaMerReactAgentConfig, run_multi_episode_rollout
from cube_harness.agents.reflection import SimpleReflectionProviderConfig
from cube_harness.core import Trajectory
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig

# Auto-load .env so OPENAI_API_KEY / OPENAI_API_BASE / etc. are visible regardless
# of how the recipe is invoked (uv run, .venv/bin/python, VS Code debugger).
# override=True so .env values win over a stale shell-exported OPENAI_API_KEY.
load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level config — what `test_recipe_defines_experiment` scans for.
# ---------------------------------------------------------------------------

benchmark_config = MINIWOB_CONFIGS["default"]
llm_config = LLMConfig(model_name="openai/gpt-4o")
# Module-level agent_config has `memory_path=None`; main() rebuilds it per-rollout with the
# appropriate path. The Experiment below is a "skeleton" that satisfies the recipe-imports guard —
# main() uses Episode for-loops (via run_multi_episode_rollout), not exp_runner.
agent_config = LaMerReactAgentConfig(
    llm_config=llm_config,
    reflection_provider=SimpleReflectionProviderConfig(),
)

exp = Experiment(
    name="lamer_miniwob",
    agent_config=agent_config,
    benchmark_config=benchmark_config,
    max_steps=10,
)


# ---------------------------------------------------------------------------
# Recipe main — sequential N-episode rollout against one task.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LaMer cross-episode rollout on MiniWob.")
    p.add_argument(
        "--task-id",
        default=None,
        help="MiniWoB task ID (default: first task in MINIWOB_CONFIGS['default']).",
    )
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--model", default=llm_config.model_name)
    p.add_argument("--max-steps", type=int, default=exp.max_steps)
    return p.parse_args(argv)


def _resolve_task(task_id: str | None):
    """Pick a single TaskConfig from the benchmark, defaulting to the first one."""
    task_configs = list(benchmark_config.get_task_configs())
    if task_id is None:
        return task_configs[0]
    match = [tc for tc in task_configs if tc.task_id == task_id]
    if not match:
        available = ", ".join(tc.task_id for tc in task_configs[:10])
        raise SystemExit(f"No task with id={task_id}. First 10: {available}")
    return match[0]


def run_rollout(
    *,
    task_config,
    experiment_dir: Path,
    rollout_idx: int,
    n_episodes: int,
    model: str,
    max_steps: int,
) -> list[Trajectory]:
    """Run one sequential rollout: N episodes against ``task_config``.

    Per-rollout output layout (recipe convention from the RFC):

        <experiment_dir>/rollouts/rollout_<idx>/
        ├── episodes/<traj_id>/...     (FileStorage V2)
        └── agent_state/lamer.json     (LaMerReactAgent memory file)
    """
    rollout_dir = experiment_dir / "rollouts" / f"rollout_{rollout_idx:03d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    memory_path = rollout_dir / "agent_state" / "lamer.json"

    rollout_agent_config = agent_config.model_copy(
        update={
            "llm_config": LLMConfig(model_name=model),
            "memory_path": memory_path,
        }
    )

    logger.info(
        "Rollout %d: task=%s, n_episodes=%d, memory=%s",
        rollout_idx,
        task_config.task_id,
        n_episodes,
        memory_path,
    )

    with benchmark_config.make() as benchmark:
        # Shared harness helper (same one PipelineRL's cube_rl rollout uses). early_stop=never so
        # this POC runs all n_episodes; persist_episode keeps each episode under rollout_dir.
        trajectories: list[Trajectory] = run_multi_episode_rollout(
            rollout_agent_config,
            task_config,
            benchmark._runtime_context,
            n_episodes,
            output_dir=rollout_dir,
            exp_name=exp.name,
            max_steps=max_steps,
            persist_episode=True,
            early_stop=lambda traj: False,
        )
    for k, traj in enumerate(trajectories):
        logger.info("Episode %d/%d done — reward=%s", k + 1, n_episodes, traj.reward_info.get("reward"))
    return trajectories


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    task_config = _resolve_task(args.task_id)
    experiment_dir = make_experiment_output_dir("lamer", "miniwob")
    trajectories = run_rollout(
        task_config=task_config,
        experiment_dir=experiment_dir,
        rollout_idx=0,
        n_episodes=args.n_episodes,
        model=args.model,
        max_steps=args.max_steps,
    )
    rewards = [t.reward_info.get("reward", 0.0) for t in trajectories]
    print()
    print(f"Rollout under {experiment_dir / 'rollouts' / 'rollout_000'}")
    print(f"  Per-episode rewards: {rewards}")
    print(f"  Final memory entries: see {experiment_dir / 'rollouts' / 'rollout_000' / 'agent_state' / 'lamer.json'}")


if __name__ == "__main__":
    main()
