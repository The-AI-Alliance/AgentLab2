"""End-to-end task-logic tests against the hermetic debug task. No credentials, no network."""

import inspect

from cube.core import Action, Observation
from cube.task import AgentView, Task

from knows_cube.debug import DebugKnowsBenchmarkConfig, DebugKnowsTask, make_debug_agent

_DOCS = "knows-debug.docs.1"
_SHEETS = "knows-debug.sheets.1"


def _task(task_id: str) -> DebugKnowsTask:
    config = next(tc for tc in DebugKnowsBenchmarkConfig().get_task_configs() if tc.metadata.id == task_id)
    return config.make()


def test_reset_is_deterministic() -> None:
    """cube test's reset-reproducibility check requires this; the real task cannot satisfy it."""
    first, _ = _task(_DOCS).reset()
    second, _ = _task(_DOCS).reset()
    assert first.model_dump() == second.model_dump()


def test_submit_ends_the_episode_and_scores_one() -> None:
    task = _task(_DOCS)
    task.reset()
    assert task.finished() is False
    task.step(Action(name="submit_work", arguments={"summary": "debug-complete"}))
    assert task.finished() is True
    reward, info = task.evaluate()
    assert reward == 1.0
    assert info["episode_end"] == "submitted"


def test_truncated_episode_is_still_graded() -> None:
    """The finalize() replacement: evaluate() must work with no submit at all."""
    task = _task(_DOCS)
    task.reset()
    reward, info = task.evaluate()
    assert reward == 0.5  # doc_id_present scores, submitted does not
    assert info["episode_end"] == "truncated"


def test_report_infeasible_ends_the_episode() -> None:
    task = _task(_DOCS)
    task.reset()
    task.step(Action(name="report_infeasible", arguments={"explanation": "nope"}))
    assert task.finished() is True
    _, info = task.evaluate()
    assert info["episode_end"] == "reported_infeasible"
    assert info["infeasible_reason"] == "nope"


def test_evaluate_is_idempotent() -> None:
    task = _task(_SHEETS)
    task.reset()
    task.step(Action(name="submit_work", arguments={"summary": "debug-complete"}))
    assert task.evaluate() == task.evaluate()


def test_reset_clears_state_between_episodes() -> None:
    """A reused task must not carry a previous episode's verdict."""
    task = _task(_SHEETS)
    task.reset()
    task.step(Action(name="submit_work", arguments={"summary": "debug-complete"}))
    assert task.evaluate()[0] == 1.0
    task.reset()
    assert task.finished() is False
    assert task.evaluate()[0] == 0.5


def test_close_is_idempotent() -> None:
    task = _task(_SHEETS)
    task.reset()
    task.close()
    task.close()


def test_eval_schema_keys_are_present() -> None:
    """The flat eval.* schema is what downstream reporting consumes."""
    task = _task(_DOCS)
    task.reset()
    task.step(Action(name="submit_work", arguments={"summary": "debug-complete"}))
    _, info = task.evaluate()
    for key in ("eval.score_result", "eval.score_total", "eval.score_fraction", "eval.n_checkpoints"):
        assert key in info, key
    assert info["eval.cp1_name"] == "submitted"


def test_post_action_records_a_url() -> None:
    """Regression: browsing history must accumulate via the per-action hook.

    The 94 evaluators that accept `browsing_history` grade against this list, so
    under-recording it degrades scores silently.

    Drives `_post_action` directly. Going through `step(submit_work)` instead
    proves nothing: `evaluate()` records a URL itself, so the list grows even with
    this override reverted to the base no-op.
    """
    task = _task(_DOCS)
    task.reset()
    before = len(task._visited_urls)
    task._post_action(Observation.from_text("[debug] no browser attached"))
    assert len(task._visited_urls) > before, "_post_action did not record a URL"


def test_hook_is_one_cube_standard_actually_calls() -> None:
    """The per-action hook we record URLs in must still exist and still be invoked.

    Both views must call it: gym `Task.step` and `AgentView.execute_action` (the
    path cube_harness episodes take). If upstream drops or renames either call,
    browsing history silently stops accumulating and 94 evaluators grade against
    a one-element list — a reward drop with no error to explain it.
    """
    assert hasattr(Task, "_post_action"), "base lost _post_action; revisit the URL-recording hook"
    assert "_post_action" in inspect.getsource(Task.step), "gym view no longer calls _post_action"
    assert "_post_action" in inspect.getsource(AgentView.execute_action), "agent view no longer calls _post_action"


def test_validate_per_step_is_pinned_off() -> None:
    """Per-step evaluation would bill a full Gemini grade per action and freeze the cached verdict."""
    task = _task(_DOCS)
    assert task.validate_per_step is False


def test_debug_agent_submits_the_winning_phrase() -> None:
    action = make_debug_agent(_DOCS)(None, [])  # type: ignore[arg-type]
    assert action.name == "submit_work"
    assert "debug-complete" in action.arguments["summary"]
