"""Gold-patch oracle baseline for SWE-bench Live.

Imports require cube-harness on the path; install via the workspace or
``pip install cube-harness`` once published.
"""

from swebench_live_cube.gold_patch.agent import GoldPatchAgent, GoldPatchAgentConfig
from swebench_live_cube.gold_patch.recipe import (
    extract_solvable,
    intersect_solvable,
    run_gold_baseline,
    run_once,
)

__all__ = [
    "GoldPatchAgent",
    "GoldPatchAgentConfig",
    "extract_solvable",
    "intersect_solvable",
    "run_gold_baseline",
    "run_once",
]
