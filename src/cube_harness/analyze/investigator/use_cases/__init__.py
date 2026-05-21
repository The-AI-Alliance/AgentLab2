"""Use-case catalog — one subdirectory per investigator recipe.

Each use case is a self-contained directory:

    use_cases/<name>/
    ├── __init__.py
    ├── recipe.py          # exports RECIPE: InvestigatorRecipe
    ├── SKILL.md           # skill description for the Auto-CUBE orchestrator
    └── scripts/           # optional helpers

This module walks the subdirectories on import, imports each `recipe.py`'s
`RECIPE` constant, and assembles `RECIPE_CATALOG: dict[str, InvestigatorRecipe]`.

Adding a use case is a one-directory PR: drop a new subdirectory in here, and
it shows up in the catalog at next import. No registration step.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

from cube_harness.analyze.investigator.recipe import InvestigatorRecipe

logger = logging.getLogger(__name__)


def _build_catalog() -> dict[str, InvestigatorRecipe]:
    """Walk subpackages of `use_cases/`, importing each `recipe.py`'s `RECIPE`.

    Sub-packages without a `recipe.py` (or a `recipe.py` that doesn't expose a
    `RECIPE` constant) are skipped with a warning — keeps the import resilient
    to half-finished work-in-progress dirs.
    """
    catalog: dict[str, InvestigatorRecipe] = {}
    pkg_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(pkg_path)]):
        if not module_info.ispkg:
            continue
        recipe_module_name = f"{__name__}.{module_info.name}.recipe"
        try:
            module = importlib.import_module(recipe_module_name)
        except ImportError as e:
            logger.warning("use_cases: skipping %s — recipe import failed: %s", module_info.name, e)
            continue
        recipe = getattr(module, "RECIPE", None)
        if not isinstance(recipe, InvestigatorRecipe):
            logger.warning("use_cases: skipping %s — `RECIPE` is missing or not a InvestigatorRecipe", module_info.name)
            continue
        if recipe.name != module_info.name:
            logger.warning(
                "use_cases: recipe.name=%r does not match directory %r — using directory name as catalog key",
                recipe.name,
                module_info.name,
            )
        catalog[module_info.name] = recipe
    return catalog


# Built once on import. Treated as immutable — see EX-002 in the constitution
# (no global mutable state); the dict is populated here and never mutated again.
RECIPE_CATALOG: dict[str, InvestigatorRecipe] = _build_catalog()


__all__ = ["RECIPE_CATALOG"]
