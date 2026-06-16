"""Reference recipe: Genny on SWE-Gym Lite.

This file IS the config — copy it and edit the values. It is not a CLI.
- Agent: a canonical config by name; tweak attributes after binding.
- Benchmark: SWE-Gym Lite. For a non-canonical subset, chain
  `.subset_from_glob(...)` / `.subset_from_list(...)`.
- Infra: named, from ~/.cube/infra.py; "local" works with zero setup.
"""

from swegym_lite_cube import SWEGYM_LITE_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

agent = GENNY_CONFIGS["swe"]
agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)

exp = Experiment(
    name="genny-swegym-lite",
    agent_config=agent,
    benchmark_config=SWEGYM_LITE_CONFIGS["default"],
    infra=INFRA_CONFIGS["local"],
    max_steps=150,
    max_cost_usd=2.0,
)

if __name__ == "__main__":
    run(exp)
