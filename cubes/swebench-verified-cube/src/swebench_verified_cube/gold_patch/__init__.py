"""Gold-patch oracle baseline for SWE-bench Verified.

Imports require cube-harness on the path; install via the workspace or
``pip install cube-harness`` once published.
"""

from swebench_verified_cube.gold_patch.agent import GoldPatchAgent, GoldPatchAgentConfig
from swebench_verified_cube.gold_patch.solvable import extract_solvable, intersect_solvable

__all__ = [
    "GoldPatchAgent",
    "GoldPatchAgentConfig",
    "extract_solvable",
    "intersect_solvable",
]
