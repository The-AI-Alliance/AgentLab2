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
"""Reference recipe: Genny on MiniWoB. The simplest end-to-end example.

This file IS the config — copy it and edit the values. It is not a CLI;
`run()` provides a fixed generic CLI (see `cube_harness.recipe`).
"""

from miniwob_cube import MINIWOB_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

agent = GENNY_CONFIGS["default"]
agent.llm_config = LLMConfig(model_name="gpt-5.4-mini", temperature=1.0)

exp = Experiment(
    name="miniwob",
    agent_config=agent,
    benchmark_config=MINIWOB_CONFIGS["default"],
    max_steps=10,
)

if __name__ == "__main__":
    run(exp)
