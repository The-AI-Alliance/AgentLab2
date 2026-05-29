"""Recipe guards: imports resolve, and each recipe builds a valid Experiment.

1. `test_recipe_local_imports_resolve` — static AST walk: every
   `from <local pkg> import Y` in a recipe resolves. Catches the class of bug
   that broke #381 (a recipe importing a symbol another PR deleted).
2. `test_recipe_defines_experiment` — executes the recipe module and asserts
   it builds an `Experiment` (or a `dict[str, Experiment]`). Catches
   config-schema drift: a renamed config field fails here, not at next run.

No Docker, no network, no LLM (recipes do their run() under `__main__`).
Recipes whose cube/tool deps aren't installed in this env are skipped, not
failed — cube CI / local dev with the cube installed picks up the assertion.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

from cube_harness.experiment import Experiment

# Package prefixes owned by this repo. Imports of these MUST resolve.  Imports
# from external packages (cube_infra_*, anthropic, etc.) are best-effort and
# may live behind optional dependency groups — skip when unavailable.
LOCAL_PACKAGE_PREFIXES: tuple[str, ...] = (
    "cube_harness",
    # Cubes under cubes/* (workspace packages installed by `make install`).
    "arithmetic_cube",
    "browsercomp",
    "miniwob",
    "osworld_cube",
    "swebench_live_cube",
    "swebench_verified_cube",
    "terminalbench2_cube",
    "webarena_verified_cube",
    "waa_cube",
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _recipes() -> list[Path]:
    """Return all recipe .py files under recipes/, excluding venvs and dunders."""
    return [
        p
        for p in REPO_ROOT.glob("recipes/**/*.py")
        if "venv" not in p.parts and not p.name.startswith("_") and p.name != "__init__.py"
    ]


def _iter_import_targets(path: Path) -> list[tuple[str, str]]:
    """Return [(module, name), ...] for every `from X import Y` in the file."""
    tree = ast.parse(path.read_text())
    targets: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                targets.append((node.module, alias.name))
    return targets


def _is_local(module: str) -> bool:
    root = module.split(".", 1)[0]
    return root in LOCAL_PACKAGE_PREFIXES


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_recipe_local_imports_resolve(recipe: Path) -> None:
    """Every `from <local pkg> import Y` in the recipe must resolve."""
    for module, name in _iter_import_targets(recipe):
        if not _is_local(module):
            continue
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as e:
            # Local package unavailable in this env (e.g. tests.yml CI doesn't
            # install workspace cubes) — no regression to assert here.
            pytest.skip(f"{module} not installed in this env ({e})")
        assert hasattr(mod, name), f"{recipe.relative_to(REPO_ROOT)}: `from {module} import {name}` — name not found"


def _runnable_recipes() -> list[Path]:
    # *_template.py (e.g. infra_template.py → ~/.cube/infra.py) are copy-me
    # templates, not runnable recipes — they define no Experiment by design.
    return [p for p in _recipes() if not p.name.endswith("_template.py")]


@pytest.mark.parametrize("recipe", _runnable_recipes(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_recipe_defines_experiment(recipe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing the recipe builds at least one Experiment (var or dict value)."""
    monkeypatch.setattr("cube_harness.experiment.make_experiment_output_dir", lambda *a, **k: tmp_path)
    spec = importlib.util.spec_from_file_location(recipe.stem, recipe)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        # Recipe's cube/tool dep not installed here — the AST guard above
        # already covers import-name resolution when it is installed.
        pytest.skip(f"recipe dependency not installed: {e.name}")
    exps: list[Experiment] = []
    for v in vars(module).values():
        if isinstance(v, Experiment):
            exps.append(v)
        elif isinstance(v, dict):
            exps.extend(x for x in v.values() if isinstance(x, Experiment))
    assert exps, f"{recipe.relative_to(REPO_ROOT)} defines no Experiment (a var or a dict[str, Experiment])"
