"""Hint-harvesting investigator: extract task-specific hints from failed episodes."""

from cube_harness.analyze.investigator.use_cases.hinter.recipe import (
    RECIPE,
    HinterOutput,
    TaskHint,
)

__all__ = ["RECIPE", "HinterOutput", "TaskHint"]
