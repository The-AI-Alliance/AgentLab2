"""Smoke tests for tau2_cube.

Run with:  pytest tests/
"""

import tau2_cube.debug as _debug_mod
from cube.testing import assert_debug_tasks_reward_one
from tau2_cube.benchmark import CubeBenchmarkConfig


def test_benchmark_metadata() -> None:
    """Benchmark metadata is valid, non-empty, and has no unfilled placeholders."""
    meta = CubeBenchmarkConfig.benchmark_metadata
    assert meta.name, "benchmark name must not be empty"
    assert meta.version, "benchmark version must not be empty"
    assert meta.description, "benchmark description must not be empty"
    assert "TODO" not in meta.description, "benchmark description still contains a TODO placeholder"


def test_task_metadata_keys_match() -> None:
    """Every task_metadata key matches its TaskMetadata.id field."""
    for key, meta in CubeBenchmarkConfig.task_metadata.items():
        assert key == meta.id, f"Key {key!r} does not match TaskMetadata.id {meta.id!r}"


def test_debug_tasks() -> None:
    """Every debug task completes with reward == 1.0 (no LLM required).

    To make this test pass:
      1. Fill in _TASK_ACTIONS in debug.py with action sequences per task ID.
      2. Ensure each sequence drives the task to done=True, reward=1.0.
    """
    assert_debug_tasks_reward_one(_debug_mod)
