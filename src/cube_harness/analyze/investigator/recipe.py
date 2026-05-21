"""`InvestigatorRecipe` — what the investigator is asked to do.

A recipe bundles the prompts, the per-recipe `OutputModel`, and a few knobs
(model, allowed tools, audit toggle). It does **not** carry a driver or a
selector — those are call-time arguments because transport choice and episode
filtering are orthogonal to the analysis being requested.

Recipes live as Python files (`use_cases/<name>/recipe.py`) so that prompts,
schema, and helpers stay co-located and Go-to-Definition works.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from cube.core import TypedBaseModel
from pydantic import ConfigDict, field_serializer, field_validator

from cube_harness.eval_log import BaseFindings


def _serialize_output_model(output_model: type[TypedBaseModel]) -> str:
    """Encode the output_model class as `module.qualname` for JSON dumps."""
    return f"{output_model.__module__}.{output_model.__qualname__}"


def _deserialize_output_model(value: Any) -> type[TypedBaseModel]:
    """Decode `module.qualname` back to a class object via importlib."""
    if isinstance(value, type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"output_model must be a class or 'module.qualname' string, got {type(value).__name__}")
    module_name, _, qualname = value.rpartition(".")
    if not module_name:
        raise ValueError(f"output_model string {value!r} has no module qualifier")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type) or not issubclass(obj, TypedBaseModel):
        raise TypeError(f"{value!r} did not resolve to a TypedBaseModel subclass")
    return obj


class InvestigatorRecipe(TypedBaseModel):
    """A use-case-specific investigator configuration.

    The runtime records which recipe was used in `investigation_metadata.json`; that's the
    JSON side. The recipe object itself stays in Python so prompts and the
    `output_model` class can be edited and Go-to-Definition'd in an IDE.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    system_prompt: str
    user_prompt_template: str
    output_model: type[TypedBaseModel]
    model: str = "claude-sonnet-4-6"
    allowed_tools: tuple[str, ...] = ("Read", "Glob", "Grep", "Bash")
    permission_mode: Literal["bypassPermissions", "ask"] = "bypassPermissions"
    audit: bool = False

    @field_serializer("output_model")
    def _serialize_output_model_field(self, value: type[TypedBaseModel]) -> str:
        return _serialize_output_model(value)

    @field_validator("output_model", mode="before")
    @classmethod
    def _validate_output_model_field(cls, value: Any) -> type[TypedBaseModel]:
        return _deserialize_output_model(value)


def get_default_recipe() -> InvestigatorRecipe:
    """Lazy accessor for the default `general_blame` recipe.

    Defined as a function rather than a module-level constant to avoid an import
    cycle: `use_cases/general_blame/recipe.py` imports `InvestigatorRecipe` from this
    module. Callers that want the default ask via `get_default_recipe()`.

    `BaseFindings` is re-exported here for convenience: every recipe's
    `OutputModel` extends it, so `from .recipe import BaseFindings` is the
    natural import path inside a use case.
    """
    from cube_harness.analyze.investigator.use_cases.general_blame.recipe import RECIPE

    return RECIPE


__all__ = ["InvestigatorRecipe", "BaseFindings", "get_default_recipe"]
