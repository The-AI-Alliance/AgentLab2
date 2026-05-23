"""Pins the verifier-result resolution invariant in `TerminalBench2Task`.

The upstream `test.sh` writes a structured CTRF report (`pytest --ctrf`). We
prefer it over regex-parsing stdout because heavy eval-time `apt`/`uvx`
installs routinely truncate pytest's text summary out of the captured stdout
(observed on pytorch-model-cli / sam-cell-seg / mcmc-sampling-stan, where even
the *reference* solution scored total=0). `verifier_ran` distinguishes a real
graded result from a verifier that never produced one (deps wouldn't install /
tests didn't collect) — so a `reward=0` of the latter kind isn't conflated with
a genuine graded failure.
"""

from __future__ import annotations

import json

from terminalbench2_cube.task import TerminalBench2Task


def _bare_task() -> TerminalBench2Task:
    """A method-only TerminalBench2Task: `_parse_verifier_results` uses no
    instance state, so `__new__` (no Pydantic init / container) suffices."""
    return TerminalBench2Task.__new__(TerminalBench2Task)


def test_ctrf_with_tests_is_graded_and_marks_ran() -> None:
    ctrf = json.dumps(
        {
            "results": {
                "tests": [
                    {"name": "test_a", "status": "passed"},
                    {"name": "test_b", "status": "failed"},
                    {"name": "test_c", "status": "skipped"},  # skipped counts as pass
                ]
            }
        }
    )
    results, ran = _bare_task()._parse_verifier_results(ctrf, "stdout ignored when CTRF present")
    assert ran is True
    assert results == {"test_a": "passed", "test_b": "failed", "test_c": "passed"}


def test_ctrf_zero_tests_marks_not_ran() -> None:
    """mcmc-sampling-stan case: pytest ran but collected 0 tests (setup failed)."""
    ctrf = json.dumps({"results": {"summary": {"tests": 0}, "tests": []}})
    results, ran = _bare_task()._parse_verifier_results(ctrf, "apt/uvx logs, no pytest summary")
    assert results == {}
    assert ran is False


def test_ctrf_survives_truncated_stdout() -> None:
    """The whole point: a valid CTRF report grades correctly even when stdout
    is just the (truncated) install log with no pytest summary."""
    ctrf = json.dumps({"results": {"tests": [{"name": "t1", "status": "passed"}]}})
    truncated_stdout = "Get:1 http://deb.debian.org/debian ... Building dependency tree ..."
    results, ran = _bare_task()._parse_verifier_results(ctrf, truncated_stdout)
    assert ran is True
    assert results == {"t1": "passed"}


def test_no_ctrf_falls_back_to_stdout() -> None:
    results, ran = _bare_task()._parse_verifier_results("", "===== 3 passed, 1 failed in 2.1s =====")
    assert ran is True
    assert sum(1 for v in results.values() if v == "passed") == 3


def test_no_ctrf_no_results_marks_not_ran() -> None:
    """sam-cell-seg case: install failed before pytest → no CTRF, no parseable stdout."""
    stdout = "ERROR: No matching distribution found for torch==2.5.1+cpu"
    results, ran = _bare_task()._parse_verifier_results("", stdout)
    assert results == {}
    assert ran is False


def test_malformed_ctrf_falls_back_to_stdout() -> None:
    results, ran = _bare_task()._parse_verifier_results("{not valid json", "===== 5 passed in 1s =====")
    assert ran is True
    assert sum(1 for v in results.values() if v == "passed") == 5
