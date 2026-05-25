"""The investigator must never silently lose a finding when the model finishes its
analysis but emits no parseable/complete structured verdict: it re-prompts (no tools)
to repair the verdict, and falls back to a LOUD sentinel rather than an empty record."""

from __future__ import annotations

import asyncio
from typing import Any

from cube_harness.analyze.investigator.agent_driver import DriverResult
from cube_harness.analyze.investigator.core import (
    _parse_findings,
    _run_with_verdict_retry,
    _UnusableVerdict,
    _verdict_failure_sentinel,
)
from cube_harness.analyze.investigator.recipe import InvestigatorRecipe
from cube_harness.eval_log import BaseFindings

# A full analysis but NO structured verdict (the kv-store-grpc failure mode).
PROSE_ONLY = "I walked the episode. The agent built the server, then the infra crashed. (no verdict block)"
# A valid, complete verdict.
VALID_VERDICT = """Here is my verdict.
```json
{"analysis": "a", "evidence": [], "summary": "s", "outcome": "success",
 "primary_blame": "none", "primary_blame_confidence": 0, "other_blames": [],
 "hypothesis": "h", "hypothesis_confidence": 0}
```"""

RECIPE = InvestigatorRecipe(
    name="t",
    system_prompt="sys",
    user_prompt_template="{trajectory_id}",
    output_model=BaseFindings,
    max_verdict_retries=2,
)
BASE_KWARGS: dict[str, Any] = dict(system_prompt="sys", user_prompt="orig", allowed_tools=("Read", "Bash"))


class _ScriptedDriver:
    """AgentDriver that returns a scripted output per `run` call (repeats the last)."""

    name = "scripted"
    max_parallelism = 1

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> DriverResult:
        text = self._outputs[min(len(self.calls), len(self._outputs) - 1)]
        self.calls.append(kwargs)
        return DriverResult(output_text=text, prompt_tokens=10, completion_tokens=5)

    async def continue_session(self, **kwargs: Any) -> DriverResult:
        raise NotImplementedError


def _run(driver: _ScriptedDriver) -> BaseFindings:
    findings, _result = asyncio.run(
        _run_with_verdict_retry(driver, RECIPE, base_run_kwargs=dict(BASE_KWARGS), episode_name="ep0")
    )
    return findings


def test_parse_findings_raises_on_unusable_verdict() -> None:
    try:
        _parse_findings(PROSE_ONLY, RECIPE)
    except _UnusableVerdict:
        pass
    else:
        raise AssertionError("expected _UnusableVerdict for prose with no verdict block")
    assert _parse_findings(VALID_VERDICT, RECIPE).outcome.value == "success"


def test_usable_verdict_first_try_does_not_retry() -> None:
    driver = _ScriptedDriver([VALID_VERDICT])
    findings = _run(driver)
    assert findings.outcome.value == "success"
    assert len(driver.calls) == 1  # no repair call


def test_retry_repairs_unusable_verdict() -> None:
    driver = _ScriptedDriver([PROSE_ONLY, VALID_VERDICT])
    findings = _run(driver)
    assert findings.outcome.value == "success"
    assert len(driver.calls) == 2  # one repair re-prompt
    repair = driver.calls[1]
    assert repair["allowed_tools"] == ()  # verdict-only repair: tools stripped
    assert "PRIOR ANALYSIS" in repair["user_prompt"] and PROSE_ONLY in repair["user_prompt"]


def test_exhausted_retries_yield_loud_sentinel() -> None:
    driver = _ScriptedDriver([PROSE_ONLY])  # never repairs
    findings = _run(driver)
    # max_verdict_retries=2 ⇒ initial + 2 repairs = 3 calls
    assert len(driver.calls) == 3
    assert findings.outcome.value == "failure"
    assert findings.summary.startswith("INVESTIGATION FAILED")
    assert PROSE_ONLY in findings.analysis  # prose preserved, not an empty record


def test_sentinel_is_a_valid_finding() -> None:
    sentinel = _verdict_failure_sentinel("some analysis", "boom")
    assert isinstance(sentinel, BaseFindings)
    assert sentinel.outcome.value == "failure"
    assert "boom" in sentinel.summary
