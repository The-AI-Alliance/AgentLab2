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
"""POC recipe: LaMer-style cross-episode rollout on MiniWob.

Runs ``n_episodes`` against the SAME MiniWoB task with a long-lived
``LaMerAgent`` that reflects on every finished trajectory and folds the
reflection into its prompt for subsequent episodes.

See the ``multi-episode-rollouts`` change in ``openspec/changes/`` for the RFC.
Motivating paper: LaMer (arxiv 2512.16848).

Usage:
    .venv/bin/python recipes/lamer_miniwob.py
    .venv/bin/python recipes/lamer_miniwob.py --task-id click-button --n-episodes 5
"""

import argparse
import logging
import sys
from typing import Any

from cube.core import ActionSchema
from dotenv import load_dotenv
from litellm import Message
from miniwob_cube import MINIWOB_CONFIGS

from cube_harness import make_experiment_output_dir
from cube_harness.agents.react import ReactAgent, ReactAgentConfig
from cube_harness.core import AgentOutput, Trajectory
from cube_harness.llm import LLMCall, LLMConfig, Prompt
from cube_harness.rollout import Rollout, RolloutConfig

# Auto-load .env so the recipe works the same whether invoked via VS Code
# debugger, plain `.venv/bin/python recipes/lamer_miniwob.py`, or `uv run`.
# override=True is critical: by default dotenv preserves existing env vars,
# but a shell that exports OPENAI_API_KEY=<real-openai-key> would clobber the
# Azure key we want here, sending the real OpenAI key to the Azure endpoint
# (→ 401). Override forces the .env values to win.
load_dotenv(override=True)

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_PROMPT = """\
You just attempted this task. Above is the full trajectory and the final reward
(1.0 = success, 0.0 = failure).

Write a short reflection focused only on what you can ACT on next attempt:
- What specifically did you NOT do that you should have?
- What concrete action would you try first on the next attempt?

Constraints:
- Be specific to actions visible in the trajectory above. Do not speculate about
  actions you didn't take or claim that successful actions were "unnecessary" —
  if reward=1.0, lean on what worked.
- Plain prose only. No markdown headings, no numbered lists, no code blocks.
- Under 80 words."""


# ---------------------------------------------------------------------------
# LaMerAgent — ReactAgent + cross-episode memory via reflect()
# ---------------------------------------------------------------------------


class LaMerAgentConfig(ReactAgentConfig):
    """LaMer-style ReactAgent with cross-episode memory.

    ``reflect()`` sends the just-finished trajectory to the LLM and appends
    the response as a memory entry. ``choose_steps_to_render`` splices the
    memory in just after the system message so subsequent ``step()`` calls
    benefit from past lessons.
    """

    reflection_prompt: str = DEFAULT_REFLECTION_PROMPT

    @property
    def agent_name(self) -> str:
        return f"LaMer-{self.llm_config.model_name}".replace("/", "_")

    def make(self, action_set: list[ActionSchema] | None = None, **kwargs: Any) -> "LaMerAgent":
        _ = kwargs
        return LaMerAgent(config=self, tools=action_set or [])


class LaMerAgent(ReactAgent):
    name: str = "lamer_agent"
    description: str = "ReAct with cross-episode memory via reflection (LaMer)."

    def __init__(self, config: LaMerAgentConfig, tools: list[ActionSchema]) -> None:
        super().__init__(config, tools)
        self._lamer_config = config
        self.memory: list[str] = []

    def reflect(self, trajectory: Trajectory, final_reward: float) -> AgentOutput | None:
        """Send the just-finished trajectory to the LLM; append the response to memory.

        Returns an ``AgentOutput`` carrying the reflection ``LLMCall`` (with
        ``tag="reflection"``) and ``thoughts``. Rollout appends this as a synthetic
        trajectory step so the reflection is observable in XRay/cost-stats/training
        extraction. Returns ``None`` if the LLM call fails — Rollout then records
        nothing for that reflection.
        """
        transcript = _render_trajectory_for_reflection(trajectory)
        prompt_messages: list[dict | Message] = [
            {"role": "system", "content": "You are an agent reflecting on a finished task attempt."},
            {
                "role": "user",
                "content": (
                    f"Final reward: {final_reward}\n\nTrajectory:\n\n{transcript}\n\n"
                    f"{self._lamer_config.reflection_prompt}"
                ),
            },
        ]
        prompt = Prompt(messages=prompt_messages)
        try:
            response = self.llm(prompt)
        except Exception:
            logger.exception("LaMer reflection LLM call failed; skipping memory update")
            self._reset_react_state()
            return None
        text = (response.message.content or "").strip()
        if text:
            self.memory.append(text)
            logger.info(f"LaMer memory now has {len(self.memory)} entries.")
        self._reset_react_state()
        llm_call = LLMCall(
            tag="reflection",
            llm_config=self._lamer_config.llm_config,
            prompt=prompt,
            output=response.message,
            usage=response.usage,
        )
        return AgentOutput(actions=[], llm_calls=[llm_call], thoughts=text or None)

    def _reset_react_state(self) -> None:
        """Clear ReAct's per-episode history; memory (cross-episode) persists."""
        self.history = []
        self._actions_cnt = 0

    def choose_steps_to_render(self, history: list[dict | Message]) -> list[dict | Message]:
        """Inject memory entries as a user message right after the system prompt."""
        base = super().choose_steps_to_render(history)
        if not self.memory or not base:
            return base
        memory_block = self._format_memory()
        return [
            base[0],  # system message
            {"role": "user", "content": memory_block},
            {"role": "assistant", "content": "I'll keep these lessons in mind."},
            *base[1:],
        ]

    def _format_memory(self) -> str:
        entries = "\n\n".join(f"### Reflection after attempt {i + 1}\n\n{entry}" for i, entry in enumerate(self.memory))
        return f"## Lessons from previous attempts at this task\n\n{entries}"


def _render_trajectory_for_reflection(trajectory: Trajectory) -> str:
    """Compact text transcript of a trajectory for the reflection prompt.

    Renders agent thoughts (when present) and action calls; environment turns
    are summarised as ``done`` + ``reward`` to keep the prompt short.
    """
    lines: list[str] = []
    for step in trajectory.steps:
        output = step.output
        if isinstance(output, AgentOutput):
            if output.thoughts:
                lines.append(f"[thoughts] {output.thoughts[:300]}")
            if output.actions:
                rendered = ", ".join(f"{a.name}({a.arguments})" for a in output.actions)
                lines.append(f"[action] {rendered}")
            else:
                lines.append("[action] (none)")
        else:
            lines.append(f"[env] done={output.done} reward={output.reward}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level declarations — recipes follow the "Python is the config" idiom.
# These are what `tests/test_recipe_imports.py::test_recipe_defines_experiment`
# scans for; main() can override the defaults from CLI flags.
# ---------------------------------------------------------------------------

benchmark_config = MINIWOB_CONFIGS["default"]
agent_config = LaMerAgentConfig(
    llm_config=LLMConfig(
        model_name="openai/gpt-4o",
        # POC: num_retries=1 surfaces Azure throttle/auth errors immediately
        # instead of burying them in LiteLLM's 5-attempt retry loop. Raise to
        # the LLMConfig default (5) once quota issues are resolved.
        # num_retries=1,
    )
)
rollout_config = RolloutConfig(n_episodes=3, gamma=0.9, max_steps_per_episode=10)

# Pick the first task by default — the recipe operates on a single task per run.
# A more sophisticated recipe could rotate tasks at the experiment layer; LaMer
# specifically wants N attempts at the SAME task, so we stay on one here.
_default_task_config = next(iter(benchmark_config.get_task_configs()))

# Per-run output dir under ~/cube_harness_results/ — Rollout writes:
#   rollouts/<rollout_id>/rollout_record.json
#   rollouts/<rollout_id>/episode_000.json, episode_001.json, ...
_output_dir = make_experiment_output_dir("lamer", "miniwob")

lamer_rollout = Rollout(
    task_config=_default_task_config,
    agent_config=agent_config,
    config=rollout_config,
    output_dir=_output_dir,
    exp_name="lamer_miniwob",
)


# ---------------------------------------------------------------------------
# Recipe main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LaMer cross-episode rollout on MiniWob.")
    p.add_argument(
        "--task-id",
        default=None,
        help="MiniWoB task ID (default: the first task in MINIWOB_CONFIGS['default']).",
    )
    p.add_argument("--n-episodes", type=int, default=rollout_config.n_episodes)
    p.add_argument("--gamma", type=float, default=rollout_config.gamma)
    p.add_argument("--model", default=agent_config.llm_config.model_name)
    p.add_argument("--max-steps", type=int, default=rollout_config.max_steps_per_episode)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # Apply CLI overrides to the module-level defaults.
    if args.task_id is not None:
        task_configs = list(benchmark_config.get_task_configs())
        match = [tc for tc in task_configs if tc.task_id == args.task_id]
        if not match:
            available = ", ".join(tc.task_id for tc in task_configs[:10])
            raise SystemExit(f"No task with id={args.task_id}. First 10: {available}")
        lamer_rollout.task_config = match[0]

    lamer_rollout.config = RolloutConfig(
        n_episodes=args.n_episodes, gamma=args.gamma, max_steps_per_episode=args.max_steps
    )
    if args.model != agent_config.llm_config.model_name:
        lamer_rollout.agent_config = LaMerAgentConfig(llm_config=LLMConfig(model_name=args.model))

    logger.info(
        f"Running LaMer rollout: task={lamer_rollout.task_config.task_id} model={args.model} "
        f"n_episodes={args.n_episodes} gamma={args.gamma}"
    )

    # The benchmark owns the MiniWoB HTTP server lifecycle.
    benchmark = benchmark_config.make()
    with benchmark:
        lamer_rollout.runtime_context = benchmark._runtime_context
        result = lamer_rollout.run()

    print()
    print(f"Rollout {result.rollout_id} complete.")
    print(f"  Per-episode rewards: {result.per_episode_rewards}")
    print(f"  Discounted reward (γ={args.gamma}): {result.discounted_reward:.4f}")


if __name__ == "__main__":
    main()
