"""Hinter round <N> experiment config — Auto-CUBE hinter use case.

Copied from this template and edited per round. This file IS the config. NO
`# /// script` header on purpose: run it with the repo venv, not standalone uv —

    /path/to/cube-harness/.venv/bin/python <this_dir>/exp_config.py --limit 3

The hinter raises performance by applying prompt-level interventions and then
promoting the ones that generalize. This template shows the two highest-reg
knobs the harness exposes; both read from sources the hinter edits, so an
intervention proven here graduates to its source via a PR:

  * the benchmark's clarification overlay — ``BENCHMARK_HINT`` (benchmark-wide
    orientation) + per-task clarifications — from the benchmark's
    ``benchmark_clarifications.py`` sidecar, and
  * per-action description overrides — better wording for an action, before it
    is promoted into the tool's docstring.
"""

from miniwob_cube import MINIWOB_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

# --- vary per round -------------------------------------------------------

TASK_IDS = [
    # explicit subset for this round — the tasks the intervention targets
    "book-flight",
]
MODEL = "gpt-5.4-mini"  # cheap; bump only with a reason in notes.md

# --- assemble -------------------------------------------------------------

benchmark = MINIWOB_CONFIGS["default"].subset_from_list(TASK_IDS)

# (1) Fold in the benchmark's clarification overlay. No-op until the benchmark
# ships a benchmark_clarifications.py sidecar; this is where curated benchmark
# hints + per-task clarifications reach the agent.
agent = GENNY_CONFIGS["default"].with_benchmark_clarifications(benchmark)
agent.llm_config = LLMConfig(model_name=MODEL, temperature=1.0)

# (2) Test better action wording before promoting it into the tool docstring.
# Keys must name an action in the benchmark's action space (raises otherwise).
agent.description_overrides = {
    # "click": "Click an element by its numeric ref id (shown in the AX tree), not its text.",
}

exp = Experiment(
    name="auto-cube-hinter-r<N>",
    agent_config=agent,
    benchmark_config=benchmark,
    max_steps=10,
    is_official=False,  # Auto-CUBE iteration run — never a submittable evaluation.
)

if __name__ == "__main__":
    run(exp)
