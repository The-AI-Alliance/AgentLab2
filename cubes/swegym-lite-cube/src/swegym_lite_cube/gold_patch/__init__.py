"""Gold-patch oracle baseline for SWE-Gym Lite.

Imports require cube-harness on the path; install via the workspace or
``pip install cube-harness`` once published.
"""

from swegym_lite_cube.gold_patch.agent import GoldPatchAgent, GoldPatchAgentConfig
from swegym_lite_cube.gold_patch.solvable import extract_solvable, intersect_solvable

__all__ = [
    "GoldPatchAgent",
    "GoldPatchAgentConfig",
    "extract_solvable",
    "intersect_solvable",
]
