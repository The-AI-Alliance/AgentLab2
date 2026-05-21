"""Round <N> experiment config — Auto-CUBE session.

Copied from a canonical recipe, edited for this round. This file IS the
config. NO `# /// script` header on purpose: run it with the repo venv,
not standalone uv —

    /path/to/cube-harness/.venv/bin/python <this_dir>/exp_config.py --limit 3

(`uv run` auto-syncs and can clobber a hand-installed editable cube-standard;
the repo venv already has cube-harness + the cube editable.)

Edit the three things that vary per round: the task subset, the model, and
the tool/agent config. Keep the explicit task list — it documents exactly
what this round tested.
"""

from terminalbench2_cube import TERMINALBENCH2_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

# --- vary per round -------------------------------------------------------

TASK_IDS = [
    # explicit subset for this round — the tasks the hypothesis is about
    "fix-git",
]
MODEL = "gpt-5.4-mini"  # cheap; bump only with a reason in notes.md
COST_PER_TASK = 0.50

# --- assemble -------------------------------------------------------------

agent = GENNY_CONFIGS["swe"]
agent.llm_config = LLMConfig(model_name=MODEL, temperature=1.0)
agent.budget.cost_limit = COST_PER_TASK

benchmark = TERMINALBENCH2_CONFIGS["default"].subset_from_list(TASK_IDS)

exp = Experiment(
    name="auto-cube-tbench2-r<N>",
    agent_config=agent,
    benchmark_config=benchmark,
    infra=INFRA_CONFIGS["local"],
    max_steps=agent.budget.max_actions or 60,
)

if __name__ == "__main__":
    run(exp)
