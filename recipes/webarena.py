# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cube-harness",
#     "webarena-verified-cube",
# ]
#
# [tool.uv.sources]
# cube-harness = { path = "..", editable = true }
# webarena-verified-cube = { path = "../cubes/webarena-verified", editable = true }
# ///
"""Reference recipe: Genny on WebArena-Verified, infra auto-provisioned.

This file IS the config — copy it and edit the values. It is not a CLI.
Infra is named, not constructed: pick another entry from ~/.cube/infra.py
(see `cube_harness.infra`); "local" works with zero setup. To run a single
site, clone this file and chain `.subset_from_glob("sites", "*shopping*")`.
"""

from webarena_verified_cube import WEBARENA_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

agent = GENNY_CONFIGS["default"]
agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)

exp = Experiment(
    name="webarena-verified",
    agent_config=agent,
    benchmark_config=WEBARENA_CONFIGS["default"],
    infra=INFRA_CONFIGS["local"],
    max_steps=30,
)

if __name__ == "__main__":
    run(exp)
