from miniwob_cube import MINIWOB_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

TASK_IDS = [
    "click-button",
    "click-checkboxes",
    "enter-text",
    "focus-text",
    "choose-date",
    "choose-list",
    "search-engine",
    "book-flight",
]

agent = GENNY_CONFIGS["default"]
agent.llm_config = LLMConfig(model_name="claude-haiku-4-5", temperature=1.0)

benchmark = MINIWOB_CONFIGS["default"].subset_from_list(TASK_IDS)

exp = Experiment(
    name="auto-cube-miniwob-r0",
    agent_config=agent,
    benchmark_config=benchmark,
    max_steps=10,
)

if __name__ == "__main__":
    run(exp)
