"""Reward-normalisation and evaluator-dispatch tests. Pure — no network, no credentials."""

import os
from pathlib import Path
from typing import Any

from browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from browsergym.knows.task import KnowsWorkspaceTask
from pytest import MonkeyPatch

import knows_cube.evaluator as evaluator_mod
from knows_cube.benchmark import KnowsBenchmarkConfig
from knows_cube.evaluator import load_evaluator

_SUPPLIED = {"workspace_doc_id", "browsing_history", "browsing_history_list", "cached_models"}
_KNOWN_UNSUPPLIED = {"browsing_history_doc_id", "client_doc_id"}


def test_reward_is_result_over_total() -> None:
    info: dict[str, Any] = {}
    result = Result(
        checkpoints=[Checkpoint(name="a", total=4, result=3), Checkpoint(name=None, total=6, result=0)],
        total_execution_time=9.25,
    )
    reward, breakdown = KnowsWorkspaceTask._summarize_result(result, info)
    assert reward == 0.3
    assert breakdown["eval.score_result"] == 3
    assert breakdown["eval.score_total"] == 10
    assert breakdown["eval.score_fraction"] == 0.3
    assert breakdown["eval.n_checkpoints"] == 2
    assert breakdown["eval.cp1_name"] == "a"
    assert breakdown["eval.cp2_fraction"] == 0.0


def test_zero_total_does_not_divide_by_zero() -> None:
    reward, breakdown = KnowsWorkspaceTask._summarize_result(
        Result(checkpoints=[Checkpoint(name="x", total=0, result=0)], total_execution_time=None), {}
    )
    assert reward == 0.0
    assert breakdown["eval.score_fraction"] == 0.0


def test_dispatch_covers_every_evaluator_signature() -> None:
    """Every kwarg any of the 110 evaluators accepts is either supplied or knowingly skipped.

    If upstream adds a new parameter, this fails rather than silently grading on
    degraded input.
    """
    seen: set[str] = set()
    for tm in KnowsBenchmarkConfig().tasks().values():
        seen |= set(tm.evaluator_accepted_kwargs)
    unexpected = seen - _SUPPLIED - _KNOWN_UNSUPPLIED
    assert not unexpected, f"unhandled evaluator kwargs: {sorted(unexpected)}"


_STUB_EVALUATOR = """
import os
BASE_PATH = "/nonexistent-knows-root"
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/docs_11_x/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
GOLDS_DIR = os.path.join(DATA_DIR, "golds/")
GOLD_IMAGE_ORIGINAL = os.path.join(GOLDS_DIR, "original_image.png")
TASK_MD_PATH = os.path.join(TASK_DIR, "task.md")
UNRELATED = "/etc/passwd"
def grade_checkpoints(workspace_doc_id=None):
    return None
"""


def test_load_evaluator_rewrites_derived_paths(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Constants derived from TASK_DIR at import time must be rewritten too.

    21 of 110 upstream instances compute further paths (GOLD_IMAGE_ORIGINAL,
    TASK_MD_PATH, ...) from the overridden ones. Evaluators guard those reads with
    os.path.exists(), so a stale path silently skips a scoring step and drops the
    reward with no evaluation_error to distinguish it from an agent failure.
    """
    idir = tmp_path / "docs_11_x" / "instance_1"
    idir.mkdir(parents=True)
    (idir / "evaluator.py").write_text(_STUB_EVALUATOR)
    monkeypatch.setattr(evaluator_mod, "instance_dir", lambda family, instance: idir)

    module = load_evaluator("docs_11_x", 1)

    assert module.TASK_DIR == f"{idir}{os.sep}"
    assert module.GOLDS_DIR == f"{idir / 'data' / 'golds'}{os.sep}"
    assert module.GOLD_IMAGE_ORIGINAL == str(idir / "data" / "golds" / "original_image.png")
    assert module.TASK_MD_PATH == str(idir / "task.md")
    assert module.UNRELATED == "/etc/passwd", "rewrote a string unrelated to the task tree"


def test_never_supplied_params_are_recorded_as_known_issues() -> None:
    """Tasks whose evaluator declares a param we never supply must say so in metadata."""
    for tm in KnowsBenchmarkConfig().tasks().values():
        if set(tm.evaluator_accepted_kwargs) & _KNOWN_UNSUPPLIED:
            assert tm.known_issues, f"{tm.id} grades on degraded input but records no known_issues"
