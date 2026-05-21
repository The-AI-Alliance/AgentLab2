"""Render a Pydantic model as a JSON example for an LLM prompt.

The investigator, audit, and meta-analysis sub-agents each ask an LLM to produce a
JSON object that round-trips through a specific Pydantic model. Hand-writing
the JSON example in a system prompt invites drift: the live smoke run
caught Sonnet emitting `label` where the schema said `name`, because the
example we hand-wrote didn't quite match.

This module derives the prompt example directly from the Pydantic class so
the model has exactly one spec to match. The Pydantic class stays the
source of truth; the prompt section is generated from `model_fields`.

Field order from the source class is preserved in the rendered example —
which matters: models token-emit in declared order, so putting evidence /
description before the structured commitment encourages think-before-commit.
"""

from __future__ import annotations

import json
import types
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

_NoneType = type(None)


def model_to_json_example(
    model: type[BaseModel],
    indent: int = 2,
    skip: frozenset[str] = frozenset(),
) -> str:
    """Return a JSON-shaped prompt example for `model`.

    Each leaf field becomes a short string of the form `<type — description>`
    (or `<type>` when no description is set). Nested Pydantic models recurse.
    Lists show a single example element. `Literal[...]` shows the allowed values.
    Enums show their members.

    `skip` lists top-level field names that the caller fills in post-parse
    (provenance fields, runtime metrics, schema version) — the model should
    not emit these. They are omitted from the example so the prompt and the
    Pydantic class remain a single source of truth without exposing
    runtime-only fields.

    The result is suitable for embedding inside a ```json fence in a prompt.
    """
    return json.dumps(_model_example(model, skip=skip), indent=indent)


def _model_example(model: type[BaseModel], *, skip: frozenset[str] = frozenset()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name in skip:
            continue
        out[name] = _field_example(field)
    return out


def _field_example(field: FieldInfo) -> Any:
    """Build the example value for a single field from its annotation + description."""
    annotation = field.annotation
    description = field.description
    return _annotation_example(annotation, description)


def _annotation_example(annotation: Any, description: str | None) -> Any:
    """Recursive: render a type annotation into a JSON-shaped example value."""
    if annotation is None or annotation is _NoneType:
        return _leaf("null", description)

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional / X | None
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not _NoneType]
        if len(non_none) == 1:
            return _annotation_example(non_none[0], description)
        # Real union — render the first arm; the prompt's text usually disambiguates.
        return _annotation_example(non_none[0], description)

    # Literal[...]
    if origin is Literal:
        choices = " | ".join(repr(a) for a in args)
        return _leaf(f"Literal[{choices}]", description)

    # list / tuple / Sequence
    if origin in (list, tuple) or annotation in (list, tuple):
        if args:
            inner = _annotation_example(args[0], None)
        else:
            inner = "..."
        return [inner] if not isinstance(inner, (list, dict)) else [inner]

    # dict / Mapping
    if origin is dict or annotation is dict:
        if len(args) == 2:
            v = _annotation_example(args[1], None)
        else:
            v = "..."
        return {"<key>": v}

    # Nested Pydantic model
    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _model_example(annotation)
    except TypeError:
        pass

    # Enum
    try:
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            members = " | ".join(repr(m.value) for m in annotation)
            return _leaf(f"Enum[{members}]", description)
    except TypeError:
        pass

    # Primitives — int / float / str / bool — fall through to a labelled placeholder.
    type_name = getattr(annotation, "__name__", repr(annotation))
    return _leaf(type_name, description)


def _leaf(type_label: str, description: str | None) -> str:
    """Render a primitive / Literal / enum leaf as `<type — description>`."""
    if description:
        # Single-line; trim long descriptions to keep the prompt scannable.
        d = description.strip().splitlines()[0]
        if len(d) > 140:
            d = d[:137] + "..."
        return f"<{type_label} — {d}>"
    return f"<{type_label}>"


__all__ = ["model_to_json_example"]
