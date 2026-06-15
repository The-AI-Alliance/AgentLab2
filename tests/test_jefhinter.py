"""Unit tests for the JefHinter harness (``cube_harness.jefhinter``).

Fast, no LLM/server: stub the miner's LLM and use step-less trajectories. Covers
the pure pieces — scoring, the curator/dedup DB, and miner JSON parsing — that
the end-to-end run can't cheaply assert.
"""

from __future__ import annotations

from typing import Any

from cube_harness import jefhinter as jh
from cube_harness.analyze.investigator.use_cases.hinter.recipe import TaskHint
from cube_harness.core import Trajectory


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
    assert miner.mine([_traj("t", 1.0)]) == []


def test_miner_handles_empty_hint_list() -> None:
    miner = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256))
    miner._llm = _FakeLLM('```json\n{"hints": []}\n```')  # type: ignore[assignment]
    assert miner.mine([_traj("t", 0.0)]) == []


def test_hint_has_literal_flags_instance_specific_and_passes_general() -> None:
    # Overfit examples from the audit -> instance-specific, must be rejected.
    overfit = [
        "enter the value '9'",
        "type 'KELI' into id 'tt'",
        "focus bid=41",
        "click element [123]",
        "the answer is 47",
    ]
    for text in overfit:
        assert jh.hint_has_literal(text), f"expected literal flagged: {text!r}"
    # Good generalizable hints -> method, not value, must pass.
    general = [
        "click the header to expand before the arrow icon",
        "compute the arithmetic shown and type the result",
    ]
    for text in general:
        assert not jh.hint_has_literal(text), f"expected general hint to pass: {text!r}"
    assert not jh.hint_has_literal("")


def test_miner_reject_literals_drops_overfit_hint() -> None:
    fenced = (
        '```json\n{"hints": [{"task_id": "t", "hint_type": "task_specific", '
        '"text": "Enter the value \'9\' into the input.", "rationale": "r", "confidence": 4}]}\n```'
    )
    # Default miner keeps the (overfit) hint; reject_literals drops it (-> no hint, same as none).
    keep = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256))
    keep._llm = _FakeLLM(fenced)  # type: ignore[assignment]
    assert len(keep.mine([_traj("t", 0.0)])) == 1

    drop = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256), reject_literals=True)
    drop._llm = _FakeLLM(fenced)  # type: ignore[assignment]
    assert drop.mine([_traj("t", 0.0)]) == []


def test_miner_general_prompt_selects_general_system_prompt() -> None:
    default_miner = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256))
    assert default_miner._system_prompt == jh.MINER_SYSTEM_PROMPT
    general_miner = jh.HintMiner(jh.build_llm("m", "http://localhost:1/v1", 0.5, 256), general_prompt=True)
    assert general_miner._system_prompt == jh.MINER_SYSTEM_PROMPT_GENERAL


def test_miniwob_benchmark_seed_threads_to_task_configs() -> None:
    """Cross-instance eval relies on MiniWobBenchmarkConfig.seed reaching every task config.

    Default 42 = the historical pin (so a no-flag eval is byte-identical to prior results);
    model_copy(update={'seed': N}) is exactly the per-rep override run_one applies.
    """
    import pytest

    miniwob = pytest.importorskip("miniwob_cube")
    base = miniwob.MINIWOB_CONFIGS["default"]
    assert base.seed == 42  # default pin

    varied = base.model_copy(update={"seed": 9001})
    varied_tcs = list(varied.get_task_configs())
    assert varied_tcs, "benchmark should yield task configs"
    assert all(tc.seed == 9001 for tc in varied_tcs)
    # original untouched (no shared-state mutation) -> still the seed-42 instance
    assert all(tc.seed == 42 for tc in base.get_task_configs())


def test_resolve_instance_seeds() -> None:
    """Recipe seed-resolver: fixed (None) by default, explicit seeds win, else auto held-out set."""
    import importlib.util
    from pathlib import Path

    import pytest

    recipe_path = Path(__file__).resolve().parents[1] / "recipes" / "jefhinter_miniwob.py"
    spec = importlib.util.spec_from_file_location("_jefhinter_miniwob_recipe", recipe_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError:
        pytest.skip("recipe deps (miniwob_cube) not installed in this env")

    resolve = mod._resolve_instance_seeds
    # default = fixed seed-42 (None -> run_one keeps the benchmark's pinned instance)
    assert resolve(False, "", 16) is None
    # --cross-instance auto-generates `repeats` consecutive held-out seeds from the base
    assert resolve(True, "", 4) == [mod.CROSS_INSTANCE_SEED_BASE + i for i in range(4)]
    # explicit --instance-seeds wins over the flag and parses/strips
    assert resolve(True, "11, 22 ,33", 4) == [11, 22, 33]
    assert resolve(False, "7", 16) == [7]


def test_run_one_applies_per_rep_instance_seed(monkeypatch: Any, tmp_path: Any) -> None:
    """run_one must run rep i on instance_seeds[i] (cross-instance), or the fixed seed by default.

    Monkeypatches the episode runner + storage so no browser/LLM is needed — we only assert
    the benchmark seed each rep receives.
    """
    import pytest

    miniwob = pytest.importorskip("miniwob_cube")
    bench = miniwob.MINIWOB_CONFIGS["default"].subset_from_list(["simple-arithmetic"])
    agent = jh.build_genny_agent(jh.build_llm("m", "http://localhost:1/v1", 0.7, 256), {}, 4, 5.0)

    seen: list[int] = []
    monkeypatch.setattr(jh, "run_sequentially", lambda exp, debug_limit=None: seen.append(exp.benchmark_config.seed))
    monkeypatch.setattr(jh, "run_with_ray", lambda exp, n_cpus: seen.append(exp.benchmark_config.seed))

    class _EmptyStorage:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...

        def load_all_trajectories(self) -> list:
            return []

    monkeypatch.setattr(jh, "FileStorage", _EmptyStorage)

    jh.run_one(
        "baseline",
        agent,
        bench,
        tmp_path,
        max_steps=1,
        n_parallel=1,
        debug_limit=1,
        repeats=3,
        instance_seeds=[9001, 9002, 9003],
    )
    assert seen == [9001, 9002, 9003], "each rep should run on its own held-out instance"

    seen.clear()
    jh.run_one("baseline", agent, bench, tmp_path, max_steps=1, n_parallel=1, debug_limit=1, repeats=2)
    assert seen == [42, 42], "default (no instance_seeds) keeps the fixed seed-42 instance"
