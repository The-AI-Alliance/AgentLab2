"""Profile-use-case experiment config — Auto-CUBE session.

Copied from a canonical recipe, edited for a profiling zoom-out. This file IS
the config. Run it with the repo venv (not standalone uv) —

    /path/to/cube-harness/.venv/bin/python <this_dir>/exp_config.py --limit 20

Then aggregate the run into a phase Pareto table:

    /path/to/cube-harness/.venv/bin/ch-profile <exp_dir>

The point of a profiling run is breadth on a CHEAP model — you are profiling
the harness/infra, not model quality. Leave profiling on for the whole slice;
the sampler is ~1-2 Hz and costs almost nothing.
"""

from terminalbench2_cube import TERMINALBENCH2_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.metrics.profiler import ProfileConfig
from cube_harness.recipe import run

# --- vary per round -------------------------------------------------------

TASK_IDS: list[str] = []  # empty = full benchmark; or an explicit subset to profile
MODEL = "gpt-5.4-mini"  # cheap; you're profiling infra/harness, not the model
COST_PER_TASK = 0.50
MAX_STEPS = 60

# --- assemble -------------------------------------------------------------

agent = GENNY_CONFIGS["swe"]
agent.llm_config = LLMConfig(model_name=MODEL, temperature=1.0)

benchmark = TERMINALBENCH2_CONFIGS["default"]
if TASK_IDS:
    benchmark = benchmark.subset_from_list(TASK_IDS)

# Budget caps live on Experiment, not the agent config: max_steps →
# Budget.max_agent_steps, max_cost_usd → Budget.max_cost_usd.
exp = Experiment(
    name="auto-cube-profile-r<N>",
    agent_config=agent,
    benchmark_config=benchmark,
    infra=INFRA_CONFIGS["local"],
    max_steps=MAX_STEPS,
    max_cost_usd=COST_PER_TASK,
    is_official=False,  # Auto-CUBE iteration run — never a submittable evaluation.
    # Always-on profiling: phase wall-clock + host resource sampling. Set
    # gpu=True only when a self-hosted inference server shares this host (RL).
    profile=ProfileConfig(sample_hz=2.0, gpu=False),
)

if __name__ == "__main__":
    run(exp)
