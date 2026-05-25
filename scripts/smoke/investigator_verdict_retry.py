"""SMOKE: end-to-end through the real `investigate_experiment` pipeline, the
investigator never loses a finding when the model emits no usable verdict.

Two scenarios, each on a minimal one-episode fixture with a pre-seeded context file
(so the benchmark-context sub-agent doesn't run) and a scripted driver:

  1. unusable-then-good  → the verdict is repaired via one no-tool re-prompt, and the
     real verdict is persisted to episode_record.json.
  2. always-unusable     → a LOUD sentinel finding is persisted (outcome=failure,
     summary starts "INVESTIGATION FAILED"), never a silent empty record.

    python scripts/smoke/investigator_verdict_retry.py

Prints `SMOKE OK/FAIL: investigator_verdict_retry` and exits 0/1.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import msgpack
import zstandard

from cube_harness.analyze.investigator import InvestigationConfig, investigate_experiment
from cube_harness.analyze.investigator.agent_driver import DriverResult
from cube_harness.analyze.investigator.context import INVESTIGATION_CONTEXT_FILENAME
from cube_harness.eval_log import EPISODE_RECORD_FILENAME, EpisodeRecord, UsageSummary

PROSE_ONLY = "I walked the episode: the agent read the file but the infra then crashed. (no verdict block)"
VALID_FINDINGS_JSON = json.dumps(
    {
        "analysis": "Agent ran `cat foo.py` but never attempted a fix.",
        "outcome": "failure",
        "summary": "Agent read the file but made no edit.",
        "primary_blame": "model_capability",
        "primary_blame_confidence": 4,
        "other_blames": [],
        "evidence": [{"step": 1, "quote": "cat foo.py"}],
        "hypothesis": "Add an explicit edit instruction to the system prompt.",
        "hypothesis_confidence": 3,
    }
)


class _ScriptedDriver:
    name = "scripted"
    max_parallelism = 1

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> DriverResult:
        text = self._outputs[min(len(self.calls), len(self._outputs) - 1)]
        self.calls.append(kwargs)
        return DriverResult(output_text=text, prompt_tokens=10, completion_tokens=5)

    async def continue_session(self, **kwargs: Any) -> DriverResult:
        raise NotImplementedError


def _fail(msg: str) -> None:
    print(f"SMOKE FAIL: investigator_verdict_retry — {msg}")
    sys.exit(1)


def _write_step(steps_dir: Path, name: str, payload: dict) -> None:
    raw = msgpack.packb(payload, use_bin_type=True)
    (steps_dir / name).write_bytes(zstandard.ZstdCompressor().compress(raw))


def _make_episode(root: Path, trajectory_id: str) -> Path:
    exp = root / "exp"
    ep = exp / "episodes" / trajectory_id
    (ep / "steps").mkdir(parents=True)
    _write_step(
        ep / "steps",
        "000_obs.msgpack.zst",
        {
            "output": {
                "obs": {
                    "contents": [{"data": "Fix the bug in foo.py.", "tool_call_id": None}],
                    "reward": None,
                    "done": False,
                }
            }
        },
    )
    _write_step(
        ep / "steps",
        "001_act.msgpack.zst",
        {
            "output": {
                "actions": [{"name": "Bash", "arguments": {"command": "cat foo.py"}}],
                "llm_calls": [],
                "error": None,
            }
        },
    )
    record = EpisodeRecord(
        evaluation_id="eval-1",
        sample_id=trajectory_id.split("_ep")[0],
        is_correct=False,
        score=0.0,
        num_turns=2,
        n_agent_steps=1,
        n_env_steps=1,
        usage=UsageSummary(),
        trajectory_id=trajectory_id,
        timestamp=0.0,
    )
    (ep / EPISODE_RECORD_FILENAME).write_text(record.model_dump_json(indent=2))
    (exp / INVESTIGATION_CONTEXT_FILENAME).write_text(f"# seeded\n\n```paths\nexp: {exp}\n```\n")
    return exp


def _investigate(exp: Path, tid: str, driver: _ScriptedDriver) -> Any:
    results = investigate_experiment(exp, InvestigationConfig(driver=driver, ids=[tid], synthesis_model=""))
    if tid not in results:
        _fail(f"no result for {tid}")
    findings, _meta = results[tid]
    return findings


def main() -> None:
    tid = "task1_ep0"
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        # Scenario 1: unusable → repaired on retry.
        exp1 = _make_episode(Path(d1), tid)
        driver1 = _ScriptedDriver([PROSE_ONLY, VALID_FINDINGS_JSON])
        f1 = _investigate(exp1, tid, driver1)
        if len(driver1.calls) != 2:
            _fail(f"repair: expected 2 driver calls (run + 1 repair), got {len(driver1.calls)}")
        if driver1.calls[1].get("allowed_tools") != ():
            _fail(f"repair: re-prompt should strip tools, got allowed_tools={driver1.calls[1].get('allowed_tools')!r}")
        if f1.primary_blame.value != "model_capability":
            _fail(f"repair: expected repaired verdict, got blame={f1.primary_blame.value}")
        restored = EpisodeRecord.model_validate_json((exp1 / "episodes" / tid / EPISODE_RECORD_FILENAME).read_text())
        if restored.findings is None or restored.findings.outcome.value != "failure":
            _fail("repair: repaired verdict was not persisted to episode_record.json")
        print(f"  scenario 1 (repair): {len(driver1.calls)} calls, blame={f1.primary_blame.value}, persisted ✓")

        # Scenario 2: never usable → loud sentinel (default max_verdict_retries=2 ⇒ 3 calls).
        exp2 = _make_episode(Path(d2), tid)
        driver2 = _ScriptedDriver([PROSE_ONLY])
        f2 = _investigate(exp2, tid, driver2)
        if len(driver2.calls) != 3:
            _fail(f"sentinel: expected 3 driver calls (run + 2 retries), got {len(driver2.calls)}")
        if f2.outcome.value != "failure" or not f2.summary.startswith("INVESTIGATION FAILED"):
            _fail(f"sentinel: expected loud sentinel, got outcome={f2.outcome.value} summary={f2.summary!r}")
        if PROSE_ONLY not in f2.analysis:
            _fail("sentinel: prose analysis was not preserved")
        print(f"  scenario 2 (sentinel): {len(driver2.calls)} calls, summary={f2.summary[:40]!r}... ✓")

    print("SMOKE OK: investigator_verdict_retry")


if __name__ == "__main__":
    main()
