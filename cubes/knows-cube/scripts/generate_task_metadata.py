#!/usr/bin/env python3
"""Generate ``src/knows_cube/task_metadata.json`` from the installed KNOWS package.

Developer tool. Run when the upstream KNOWS task set changes; the output is
committed and shipped as package data so end users never import
``browsergym.knows`` merely to enumerate tasks.

Reads three sources and cross-checks them:

1. the 22 concrete ``KnowsWorkspaceTask`` subclasses — class attributes only,
   never instantiated (instantiating needs BrowserGym's ABC)
2. ``<family>/instance_<n>/task.md`` for the goal text
3. ``<family>/instance_<n>/evaluator.py``, AST-parsed and never imported —
   importing 105 of the 110 authenticates against Google as a side effect

Usage::

    uv run python scripts/generate_task_metadata.py --force
"""

import ast
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from browsergym.knows import task as knows_task_module
from browsergym.knows.task import EVAL_TASKS_DIR, KnowsWorkspaceTask

logger = logging.getLogger(__name__)

# Deliberately does NOT import knows_cube. KnowsBenchmarkConfig auto-loads this
# very file at class-definition time, so importing the package here would be a
# circular bootstrap: the generator could never run against a missing or empty
# metadata file. Emitting plain dicts keeps the generator independent; the
# schema is enforced afterwards by `make metadata`, which re-imports the package
# in a fresh process, and by tests/test_metadata.py.
_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "src" / "knows_cube" / "task_metadata.json"

_TYPE_DISCRIMINATOR = "knows_cube.task.KnowsTaskMetadata"
"""Without this key every subclass field is silently dropped on load and the
entry validates as a plain TaskMetadata, with no warning."""

_NEVER_SUPPLIED = ("browsing_history_doc_id", "client_doc_id")
"""Params some evaluators declare that no caller — upstream or this cube — supplies."""

EXPECTED_TASKS = 110


def _families() -> list[type[KnowsWorkspaceTask]]:
    """All concrete KnowsWorkspaceTask subclasses, discovered by class attribute."""
    out: list[type[KnowsWorkspaceTask]] = []
    for obj in vars(knows_task_module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, KnowsWorkspaceTask)
            and obj is not KnowsWorkspaceTask
            and getattr(obj, "TASK_FAMILY_FOLDER", "")
            and getattr(obj, "AVAILABLE_INSTANCES", ())
        ):
            out.append(obj)
    return sorted(out, key=lambda c: c.TASK_FAMILY_FOLDER)


def _parse_evaluator(path: Path) -> dict[str, Any]:
    """AST-parse ``evaluator.py`` to extract its signature and checkpoint totals.

    Never imports the file. A syntax error is reported rather than raised —
    upstream ships at least one evaluator that does not parse.
    """
    if not path.exists():
        return {"gradeable": False, "issue": f"evaluator.py missing at {path}"}
    try:
        tree = ast.parse(path.read_text(), str(path))
    except SyntaxError as exc:
        return {"gradeable": False, "issue": f"evaluator.py fails to parse: {exc}"}

    kwargs: list[str] = []
    n_checkpoints = 0
    total = 0
    dynamic = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "grade_checkpoints":
            kwargs = [a.arg for a in (*node.args.args, *node.args.kwonlyargs)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Checkpoint":
            n_checkpoints += 1
            keyword = next((k for k in node.keywords if k.arg == "total"), None)
            if keyword is not None and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, int):
                total += keyword.value.value
            else:
                dynamic = True
    return {
        "gradeable": True,
        "kwargs": sorted(kwargs),
        "n_checkpoints": n_checkpoints,
        "max_points": None if dynamic else total,
        "has_dynamic_points": dynamic,
    }


def _build_metadata(cls: type[KnowsWorkspaceTask], instance: int) -> dict[str, Any]:
    """Build one task's metadata dict from its instance directory."""
    folder = cls.TASK_FAMILY_FOLDER
    instance_dir = EVAL_TASKS_DIR / folder / f"instance_{instance}"
    info = _parse_evaluator(instance_dir / "evaluator.py")

    issues: list[str] = []
    if not info["gradeable"]:
        issues.append(info["issue"])
    unsatisfied = sorted(set(info.get("kwargs", [])) & set(_NEVER_SUPPLIED))
    if unsatisfied:
        issues.append(f"evaluator declares never-supplied params: {unsatisfied}")

    id_file = instance_dir / "id.txt"
    knows_task_id = id_file.read_text().strip() if id_file.exists() else ""
    if not knows_task_id:
        issues.append("id.txt missing or empty")

    task_md = instance_dir / "task.md"
    if not task_md.exists():
        raise FileNotFoundError(f"task.md missing at {task_md}")

    n_checkpoints = info.get("n_checkpoints", 0)
    return {
        "id": f"{cls.TASK_ID_PREFIX}.{instance}",
        "split": "test",
        "abstract_description": task_md.read_text().strip(),
        # No upstream source for a step budget; scaled off checkpoint count and
        # clamped. A heuristic, and flagged as such — worth revisiting once real
        # episodes have been observed.
        "recommended_max_steps": min(60, max(20, 5 * n_checkpoints)),
        "container_config": None,
        "_type": _TYPE_DISCRIMINATOR,
        "task_family_folder": folder,
        "task_id_prefix": cls.TASK_ID_PREFIX,
        "instance_number": instance,
        "workspace_kind": cls.WORKSPACE_KIND,
        "knows_task_id": knows_task_id or None,
        "evaluator_accepted_kwargs": info.get("kwargs", []),
        "n_checkpoints": n_checkpoints,
        "max_points": info.get("max_points"),
        "has_dynamic_points": info.get("has_dynamic_points", False),
        "is_gradeable": info["gradeable"],
        "known_issues": issues,
    }


def main(
    output: Annotated[Path, typer.Option(help="Destination file.")] = _DEFAULT_OUTPUT,
    force: Annotated[bool, typer.Option(help="Regenerate even if the file already exists.")] = False,
) -> None:
    """Write src/knows_cube/task_metadata.json (110 entries)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if output.exists() and not force:
        logger.info("%s exists — pass --force to regenerate.", output)
        raise typer.Exit(0)

    metas = [_build_metadata(cls, inst) for cls in _families() for inst in cls.AVAILABLE_INSTANCES]

    if len(metas) != EXPECTED_TASKS:
        raise typer.BadParameter(f"expected {EXPECTED_TASKS} KNOWS instances, built {len(metas)}")
    ids = [m["id"] for m in metas]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise typer.BadParameter(f"duplicate task ids: {duplicates}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metas, indent=2) + "\n")

    ungradeable = [m["id"] for m in metas if not m["is_gradeable"]]
    logger.info("Wrote %d tasks to %s", len(metas), output)
    if ungradeable:
        logger.warning("%d task(s) flagged is_gradeable=False: %s", len(ungradeable), ungradeable)


if __name__ == "__main__":
    typer.run(main)
