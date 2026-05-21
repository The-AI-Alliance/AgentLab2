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
"""Reference recipe: Genny on SWE-bench Verified.

This file IS the config — copy it and edit the values. It is not a CLI.
- Agent: a canonical config by name; tweak attributes after binding.
- Benchmark: a canonical config by name. For a non-canonical subset, clone
  this file and chain `.subset_from_glob(...)` / `.subset_from_list(...)`.
- Infra: named, from ~/.cube/infra.py; "local" works with zero setup.
"""

from swebench_verified_cube import SWEBENCH_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

agent = GENNY_CONFIGS["swe"]
agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)
agent.budget.cost_limit = 2.0

exp = Experiment(
    name="genny-swebench-verified",
    agent_config=agent,
    benchmark_config=SWEBENCH_CONFIGS["default"],
    infra=INFRA_CONFIGS["local"],
    max_steps=agent.budget.max_actions or 150,
)

if __name__ == "__main__":
    run(exp)
