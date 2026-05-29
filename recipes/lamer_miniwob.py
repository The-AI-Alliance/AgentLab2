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

Runs N sequential episodes against the SAME MiniWoB task with a fresh
``LaMerAgent`` per episode. Each agent reads its `memory` list from a JSON
file at construction and writes back on ``finalize()`` — that file is what
carries lessons across the per-rollout for-loop.

Demonstrates the `multi-episode-rollouts` design
(``openspec/changes/multi-episode-rollouts/``) end-to-end:

- ``Agent.finalize(reward)`` hook runs at the end of each episode.
- File-based cross-episode memory: ``memory_path`` on ``LaMerAgentConfig``.
- Recipe-side nested output layout: ``<output_dir>/rollouts/<id>/{episodes,agent_state}/``.

Usage::

    .venv/bin/python recipes/lamer_miniwob.py
    .venv/bin/python recipes/lamer_miniwob.py --task-id click-button --n-episodes 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from cube.core import ActionSchema
from dotenv import load_dotenv
from litellm import Message
from miniwob_cube import MINIWOB_CONFIGS

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgent, ReactAgentConfig
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.episode import Episode
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMCall, LLMConfig, Prompt

# Auto-load .env so OPENAI_API_KEY / OPENAI_API_BASE / etc. are visible regardless
# of how the recipe is invoked (uv run, .venv/bin/python, VS Code debugger).
# override=True so .env values win over a stale shell-exported OPENAI_API_KEY.
load_dotenv(override=True)

logger = logging.getLogger(__name__)


DEFAULT_REFLECTION_PROMPT = """\
You just attempted this task. The final reward was {reward} (1.0 = success, 0.0 = failure).

In 2-3 sentences of plain prose, describe what worked or didn't, and what concrete
change you would try on the next attempt. Be specific to the actions you took.

No bullets, no headings, no code blocks."""


# ---------------------------------------------------------------------------
# LaMerAgent — ReactAgent + file-based cross-episode memory via finalize()
# ---------------------------------------------------------------------------


class LaMerAgentConfig(ReactAgentConfig):
    """LaMer-style ReactAgent with file-backed cross-episode memory.

    ``memory_path`` is the canonical place for the agent's memory list — a JSON
    file the agent reads on construction (``__init__``) and writes back on
    ``finalize``. The recipe sets this path per-rollout, so within one rollout
    all N episodes share the same file and accumulate lessons across attempts.
    Across rollouts, the recipe puts each rollout under its own subdir so the
    memory files don't collide.
    """

    reflection_prompt: str = DEFAULT_REFLECTION_PROMPT
    memory_path: Path | None = None

    @property
    def agent_name(self) -> str:
        return f"LaMer-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: Any) -> "LaMerAgent":
        _ = kwargs
        return LaMerAgent(config=self, tools=action_set or [])


class LaMerAgent(ReactAgent):
    """ReactAgent with cross-episode memory loaded from disk on __init__.

    Memory injection happens in ``choose_steps_to_render`` (splices a "Lessons
    from previous attempts" block after the system prompt). End-of-episode
    reflection happens in ``finalize`` (LLM call → append to memory → persist
    to ``memory_path`` → return AgentOutput so the reflection's LLMCall lands
    in the trajectory).
    """

    name: str = "lamer_agent"
    description: str = "ReAct with cross-episode memory via finalize() reflection."

    def __init__(self, config: LaMerAgentConfig, tools: list[ActionSchema]) -> None:
        super().__init__(config, tools)
        self._lamer_config = config
        self.memory: list[str] = []
        if config.memory_path is not None and config.memory_path.exists():
            try:
                data = json.loads(config.memory_path.read_text())
                self.memory = data.get("memory", [])
                logger.info("LaMer loaded %d memory entries from %s", len(self.memory), config.memory_path)
            except Exception:
                logger.exception("Failed to load memory from %s; starting empty", config.memory_path)

    def finalize(self, reward: float) -> AgentOutput | None:
        """End-of-episode reflection: LLM call → memory.append → persist → return AgentOutput.

        Returns ``None`` only if the LLM call raised (memory still persisted)
        or the LLM returned empty content (nothing to add). Otherwise returns
        an ``AgentOutput`` carrying the reflection ``LLMCall`` (tag=``"reflection"``)
        so the framework appends it to the trajectory.
        """
        prompt_messages: list[dict | Message] = [
            {"role": "system", "content": "You are an agent reflecting on a finished task attempt."},
            {
                "role": "user",
                "content": self._lamer_config.reflection_prompt.format(reward=reward),
            },
        ]
        prompt = Prompt(messages=prompt_messages)
        try:
            response = self.llm(prompt)
        except Exception:
            logger.exception("LaMer reflection LLM call failed; persisting memory unchanged")
            self._persist_memory()
            return None

        text = (response.message.content or "").strip()
        if text:
            self.memory.append(text)
            logger.info("LaMer memory now has %d entries", len(self.memory))
        self._persist_memory()

        if not text:
            return None

        llm_call = LLMCall(
            tag="reflection",
            llm_config=self._lamer_config.llm_config,
            prompt=prompt,
            output=response.message,
            usage=response.usage,
        )
        return AgentOutput(actions=[], llm_calls=[llm_call], thoughts=text)

    def _persist_memory(self) -> None:
        """Write ``self.memory`` to ``memory_path`` (creates parent dirs as needed)."""
        path = self._lamer_config.memory_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"memory": self.memory}, indent=2))
        except Exception:
            logger.exception("Failed to persist memory to %s", path)

    def choose_steps_to_render(self, history: list[dict | Message]) -> list[dict | Message]:
        """Inject memory as a user message right after the system prompt."""
        base = super().choose_steps_to_render(history)
        if not self.memory or not base:
            return base
        return [
            base[0],  # system message
            {"role": "user", "content": self._format_memory()},
            {"role": "assistant", "content": "I'll keep these lessons in mind."},
            *base[1:],
        ]

    def _format_memory(self) -> str:
        entries = "\n\n".join(f"### Reflection after attempt {i + 1}\n\n{entry}" for i, entry in enumerate(self.memory))
        return f"## Lessons from previous attempts at this task\n\n{entries}"


# ---------------------------------------------------------------------------
# Module-level config — what `test_recipe_defines_experiment` scans for.
# ---------------------------------------------------------------------------

benchmark_config = MINIWOB_CONFIGS["default"]
llm_config = LLMConfig(model_name="openai/gpt-4o")
# Module-level agent_config has `memory_path=None`; main() rebuilds it per-rollout
# with the appropriate path. The Experiment below is a "skeleton" that satisfies
# the recipe-imports guard — main() uses Episode for-loops, not exp_runner.
agent_config = LaMerAgentConfig(llm_config=llm_config)

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
        └── agent_state/lamer.json     (LaMerAgent memory file)
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

    trajectories: list[Trajectory] = []
    with benchmark_config.make() as benchmark:
        for k in range(n_episodes):
            traj = Episode(
                id=k,
                output_dir=rollout_dir,
                agent_config=rollout_agent_config,
                task_config=task_config,
                exp_name=exp.name,
                max_steps=max_steps,
                storage=None,
                runtime_context=benchmark._runtime_context,
            ).run()
            trajectories.append(traj)
            logger.info(
                "Episode %d/%d done — reward=%s",
                k + 1,
                n_episodes,
                traj.reward_info.get("reward"),
            )
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
