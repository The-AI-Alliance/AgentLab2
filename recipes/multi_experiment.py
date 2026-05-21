# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "swebench-verified-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# swebench-verified-cube = { path = "../cubes/swebench-verified", editable = true }
# ///
"""Reference recipe: a named dict of experiments.

Clone this file and modify at will — it is committed only as a worked
example. `run(experiments)` exposes `--experiment <name>` (default
"default") to pick one; e.g. `python recipes/multi_experiment.py -e swe`.
Every config lookup is an independent deep copy, so tweaking one agent
never affects the other.
"""

from swebench_verified_cube import SWEBENCH_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

default_agent = GENNY_CONFIGS["default"]
default_agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)
default_agent.budget.cost_limit = 1.0

swe_agent = GENNY_CONFIGS["swe"]
swe_agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)
swe_agent.budget.cost_limit = 1.0

experiments = {
    "default": Experiment(
        name="genny-default",
        agent_config=default_agent,
        benchmark_config=SWEBENCH_CONFIGS["default"],
        infra=INFRA_CONFIGS["local"],
    ),
    "swe": Experiment(
        name="genny-swe",
        agent_config=swe_agent,
        benchmark_config=SWEBENCH_CONFIGS["default"],
        infra=INFRA_CONFIGS["local"],
    ),
}

if __name__ == "__main__":
    run(experiments)
