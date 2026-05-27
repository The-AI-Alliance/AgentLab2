"""Recipe guards: imports resolve, and each recipe builds a valid Experiment.

1. `test_recipe_local_imports_resolve` — static AST walk: every
   `from <local pkg> import Y` in a recipe resolves. Catches the class of bug
   that broke #381 (a recipe importing a symbol another PR deleted).
2. `test_recipe_defines_experiment` — executes the recipe (as `__main__`, with
   `run()` stubbed) and asserts it yields an `Experiment` — assigned at module
   level OR passed to `run()` by a lazy builder (the gold-patch recipes build
   their Experiments inside `__main__` to avoid creating output dirs at import).
   Catches config-schema drift: a renamed config field fails here, not at next run.

Covers both the top-level `recipes/` and the per-cube recipe modules under
`cubes/*/` — the entry points downstream users hit first. No Docker, no network,
no LLM (`run()` is stubbed). Recipes whose cube/tool deps aren't installed in this
env are skipped, not failed — local dev with the cube installed asserts for real.
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
    """All recipe .py files (excluding venvs and dunders): the top-level recipes/
    plus the per-cube recipe modules (``cubes/*/.../recipe.py`` and ``*_recipe.py``).
    Those in-cube modules are the entry points downstream users invoke, yet had no
    import coverage before."""
    # In-cube recipes are package modules under src/; scoping there keeps helper
    # scripts (e.g. scripts/smoke/*_recipe.py) out.
    globs = ("recipes/**/*.py", "cubes/*/src/**/recipe.py", "cubes/*/src/**/*_recipe.py")
    found = {p for pattern in globs for p in REPO_ROOT.glob(pattern)}
    return sorted(p for p in found if "venv" not in p.parts and not p.name.startswith("_") and p.name != "__init__.py")


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


def _experiments_in(obj: object) -> list[Experiment]:
    """Experiments held directly, or as the values of a dict[str, Experiment]."""
    if isinstance(obj, Experiment):
        return [obj]
    if isinstance(obj, dict):
        return [v for v in obj.values() if isinstance(v, Experiment)]
    return []


@pytest.mark.parametrize("recipe", _runnable_recipes(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_recipe_defines_experiment(recipe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing the recipe yields at least one Experiment — assigned at module
    level, or passed to run() under __main__ (the lazy `_exp()` builder pattern)."""
    monkeypatch.setattr("cube_harness.experiment.make_experiment_output_dir", lambda *a, **k: tmp_path)
    captured: list[Experiment] = []
    # Stub run() so the recipe's `if __name__ == "__main__": run(...)` records the
    # experiments instead of launching them. `from cube_harness.recipe import run`
    # binds this stub because we patch the source attribute before exec.
    monkeypatch.setattr(
        "cube_harness.recipe.run",
        lambda *args: captured.extend(e for a in args for e in _experiments_in(a)),
    )
    # Name the module "__main__" so the recipe's `if __name__ == "__main__"` block
    # fires (loader and module name must agree, so set it on the spec, not after).
    spec = importlib.util.spec_from_file_location("__main__", recipe)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        # Recipe's cube/tool dep not installed here — the AST guard above
        # already covers import-name resolution when it is installed.
        pytest.skip(f"recipe dependency not installed: {e.name}")
    exps = list(captured)
    for v in vars(module).values():
        exps.extend(_experiments_in(v))
    assert exps, f"{recipe.relative_to(REPO_ROOT)} defines no Experiment (module-level or passed to run())"
