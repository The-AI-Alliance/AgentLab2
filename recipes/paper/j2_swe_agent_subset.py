"""J2 — agent run on a 100-task SWE-bench Verified subset, for the cross-harness regrade.

Same agent/model as ``recipes/swe_agent_recipe.py``; the only change is a
deterministic 100-task subset, because a full 500-task agent run is not
affordable for the rebuttal window.

Selection is round-robin across repositories over the id-sorted task list. That
*equalises* repos (~10 each across all 12) rather than reproducing SWE-bench
Verified's own django-heavy mix, and it is a deliberate trade: the regrade hunts
for evaluator disagreements, which arise from test-framework-specific
relaxations, so breadth of repo/test-harness coverage surfaces more of them than
a proportional sample would. The consequence is that the resulting resolve rate
is NOT comparable to a published SWE-bench Verified score — it is a parity
instrument, not a leaderboard number.

The selection is written to ``j2_subset_ids.json`` next to the run so the exact
task set is reproducible and reportable.

    .venv/bin/python recipes/paper/j2_swe_agent_subset.py --ray 16
"""

import json
from collections import defaultdict
from pathlib import Path

from swebench_verified_cube import SWEBENCH_CONFIGS

from cube_harness.agents.genny_configs import GENNY_CONFIGS
from cube_harness.experiment import Experiment
from cube_harness.infra import INFRA_CONFIGS
from cube_harness.llm import LLMConfig
from cube_harness.recipe import run

N_TASKS = 100
SUBSET_PATH = Path("/network/scratch/o/omar.younis/cube_rebuttal/j2_subset_ids.json")


def _stratified_ids(n: int) -> list[str]:
    """Round-robin across repos over id-sorted tasks — deterministic, no RNG."""
    by_repo: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(SWEBENCH_CONFIGS["default"].tasks()):
        by_repo[task_id.rsplit("-", 1)[0]].append(task_id)

    picked: list[str] = []
    round_idx = 0
    while len(picked) < n:
        added = False
        for repo in sorted(by_repo):
            if round_idx < len(by_repo[repo]):
                picked.append(by_repo[repo][round_idx])
                added = True
                if len(picked) == n:
                    break
        if not added:  # exhausted every repo
            break
        round_idx += 1
    return picked


ids = _stratified_ids(N_TASKS)
SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
SUBSET_PATH.write_text(json.dumps(ids, indent=1))

agent = GENNY_CONFIGS["swe"]
# Anthropic rather than the reference recipe's gpt-5.4-mini: the cluster only has
# an Anthropic key. Routed through LiteLLM like every other model in the harness.
# temperature is left at the LLMConfig default of 1.0 — Sonnet 5 rejects only
# *non-default* sampling values, so 1.0 passes while e.g. 0.2 would 400.
agent.llm_config = LLMConfig(model_name="anthropic/claude-sonnet-5")

exp = Experiment(
    name="j2-genny-swebench-verified-100",
    agent_config=agent,
    benchmark_config=SWEBENCH_CONFIGS["default"].subset_from_list(ids),
    infra=INFRA_CONFIGS["local"],
    max_steps=150,
    max_cost_usd=2.0,
)

if __name__ == "__main__":
    run(exp)
