import os

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

# Keep model configurable so Round 1 can run immediately with any funded provider.
MODEL_NAME = os.environ.get("AUTO_CUBE_MODEL", "claude-haiku-4-5")
MAX_STEPS = int(os.environ.get("AUTO_CUBE_MAX_STEPS", "20"))

agent = GENNY_CONFIGS["default"]
agent.llm_config = LLMConfig(model_name=MODEL_NAME, temperature=1.0)

# Intervention from Round 0: clarify ordinal interpretation for list-like tasks.
agent.system_prompt = (
    f"{agent.system_prompt}\n\n"
    "When a task says an ordinal like '5th', treat it as 1-indexed in visual reading order. "
    "Do not assume DOM attributes such as data-index or data-result are 1-indexed."
)

benchmark = MINIWOB_CONFIGS["default"].subset_from_list(TASK_IDS)

exp = Experiment(
    name="auto-cube-miniwob-r1",
    agent_config=agent,
    benchmark_config=benchmark,
    max_steps=MAX_STEPS,
)

if __name__ == "__main__":
    run(exp)
