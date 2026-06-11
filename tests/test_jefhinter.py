"""Unit tests for the JefHinter loop logic (recipes/jefhinter.py).

Fast, no LLM/server: stub the miner's LLM and use step-less trajectories. Covers
the pure pieces — scoring, the curator/dedup DB, and miner JSON parsing — that
the end-to-end run can't cheaply assert.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from cube_harness.core import Trajectory

_RECIPE_PATH = Path(__file__).resolve().parents[1] / "recipes" / "jefhinter.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("jefhinter_recipe", _RECIPE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jh = _load_module()


def _traj(task_id: str, reward: float) -> Trajectory:
    return Trajectory(id=f"{task_id}_{reward}", metadata={"task_id": task_id}, reward_info={"reward": reward})


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeLLM:
    """Stand-in for cube_harness.llm.LLM — returns a canned response."""

    def __init__(self, content: str) -> None:
        self._content = content

    def __call__(self, prompt: Any) -> _FakeResponse:
        return _FakeResponse(self._content)


def test_score_trajectories_overall_and_per_task() -> None:
    trajs = [_traj("a", 1.0), _traj("a", 0.0), _traj("b", 0.0), _traj("b", 0.0)]
    score = jh.score_trajectories(trajs)
    assert score["n_episodes"] == 4
    assert score["per_task"]["a"] == 0.5
    assert score["per_task"]["b"] == 0.0
    assert score["overall"] == 0.25


def test_hint_db_dedup_cap_and_format() -> None:
    from cube_harness.analyze.investigator.use_cases.hinter.recipe import TaskHint

    db = jh.HintDB(max_per_task=2)

    def h(text: str) -> TaskHint:
        return TaskHint(rationale="r", hint_type="task_specific", task_id="t", text=text, confidence=3)

    assert db.add([h("use the funnel icon")]) == 1
    assert db.add([h("use the funnel icon")]) == 0  # exact dedup
    assert db.add([h("click the column header")]) == 1
    assert db.add([h("open the filter panel")]) == 1  # exceeds cap=2
    assert db.size == 2  # capped, oldest dropped
    rendered = db.as_task_hints()["t"]
    assert "open the filter panel" in rendered and rendered.startswith("Hints learned")


def test_miner_parses_hint_and_injects() -> None:
    fenced = (
        '```json\n{"hints": [{"task_id": "workarena.servicenow.sort-incident-list", '
        '"hint_type": "task_specific", "text": "Click the column header, not a separate sort button.", '
        '"rationale": "the failed run searched for a sort button", "confidence": 4}]}\n```'
    )
    miner = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256))
    miner._llm = _FakeLLM(fenced)  # type: ignore[assignment]

    hints = miner.mine([_traj("workarena.servicenow.sort-incident-list", 0.0)])
    assert len(hints) == 1
    assert "column header" in hints[0].text

    db = jh.HintDB()
    db.add(hints)
    injected = db.as_task_hints()["workarena.servicenow.sort-incident-list"]
    assert "column header" in injected


def test_miner_skips_when_no_failures() -> None:
    miner = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256))
    miner._llm = _FakeLLM('```json\n{"hints": []}\n```')  # type: ignore[assignment]
    # only a passing trajectory -> nothing to mine
    assert miner.mine([_traj("t", 1.0)]) == []


def test_miner_handles_empty_hint_list() -> None:
    miner = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256))
    miner._llm = _FakeLLM('```json\n{"hints": []}\n```')  # type: ignore[assignment]
    assert miner.mine([_traj("t", 0.0)]) == []
