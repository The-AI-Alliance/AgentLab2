"""Scaffold × model matrix on MiniWoB — holds the benchmark and tool stack fixed.

Two questions the paper's single-scaffold table cannot answer:

  * **Scaffold sensitivity.** Does the ranking survive a change of agent loop?
    Genny (rolling-summary context management) and ReAct (think/act transcript)
    are run over the identical 125-task set with the identical BrowserGym tool
    stack, so any difference is attributable to the scaffold alone.
  * **Open-weight coverage.** The RL post-training motivation needs at least one
    model with updatable weights. Qwen3-VL-235B runs through the same LiteLLM
    gateway as the closed models; MiniWoB's observation is html+screenshot, so
    the open-weight model has to be vision-capable for the comparison to hold
    the tool stack constant.

Every cell is the same benchmark, same tool config, same step cap — only the
agent scaffold and the model change.

    # one cell, in-process, first 5 tasks (smoke)
    .venv/bin/python recipes/paper/miniwob_scaffold_model_matrix.py -e genny-gpt5mini --limit 5

    # one cell, full 125 tasks, 4 Ray workers
    .venv/bin/python recipes/paper/miniwob_scaffold_model_matrix.py -e react-gpt5mini --ray 4
"""

import os

from miniwob_cube import MINIWOB_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.agents.react_configs import REACT_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

MAX_STEPS = 10  # MiniWoB episodes are short; matches the paper's MiniWoB runs

# Model ids as LiteLLM resolves them. Two routes to an open-weight model: a
# hosted gateway (OpenRouter, needs OPENROUTER_API_KEY) or a self-served vLLM
# endpoint (the `qwenlocal` cell — set CUBE_HARNESS_OPENWEIGHT_BASE_URL). The
# self-served route is the one that composes with the RL rollout service, which
# needs weights you control anyway.
OPENWEIGHT_BASE_URL = os.getenv("CUBE_HARNESS_OPENWEIGHT_BASE_URL", "http://localhost:8000/v1")
OPENWEIGHT_MODEL = os.getenv("CUBE_HARNESS_OPENWEIGHT_MODEL", "hosted_vllm/Qwen/Qwen3-VL-30B-A3B-Instruct")

MODELS: dict[str, str] = {
    "gpt5mini": "gpt-5.4-mini",
    "qwen3vl": "openrouter/qwen/qwen3-vl-235b-a22b-instruct",
    "qwenlocal": OPENWEIGHT_MODEL,
}


def _llm(model: str) -> LLMConfig:
    """Point self-served models at the local endpoint; leave gateways to LiteLLM."""
    if model.startswith("hosted_vllm/"):
        return LLMConfig(
            model_name=model,
            temperature=1.0,
            api_base=OPENWEIGHT_BASE_URL,
            api_key=os.getenv("CUBE_HARNESS_LLM_API_KEY", "EMPTY"),
        )
    return LLMConfig(model_name=model, temperature=1.0)


def _genny(model: str) -> object:
    agent = GENNY_CONFIGS["default"]
    agent.llm_config = _llm(model)
    return agent


def _react(model: str) -> object:
    agent = REACT_CONFIGS["default"]
    agent.llm_config = _llm(model)
    return agent


SCAFFOLDS = {"genny": _genny, "react": _react}

EXPERIMENTS: dict[str, Experiment] = {
    f"{scaffold}-{model_key}": Experiment(
        name=f"miniwob-{scaffold}-{model_key}",
        agent_config=build(model_id),  # type: ignore[arg-type]
        benchmark_config=MINIWOB_CONFIGS["default"],
        max_steps=MAX_STEPS,
    )
    for scaffold, build in SCAFFOLDS.items()
    for model_key, model_id in MODELS.items()
}
EXPERIMENTS["default"] = EXPERIMENTS["genny-gpt5mini"]

if __name__ == "__main__":
    run(EXPERIMENTS)
