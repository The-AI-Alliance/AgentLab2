"""Event decoding: TrajectoryView → readable transcript text files.

Reads the trajectory via `FileStorage.load_episode` so the same code
path handles both layouts:

  - new (events/): LLMCallEvent / ToolCallEvent / EvaluationEvent /
    AgentErrorEvent — written by Episode after the agent-owns-loop RFC.
  - legacy (steps/): TrajectoryStep with AgentOutput | EnvironmentOutput
    — older episodes pre-dating the event-stream model.

Storage's `_events_to_legacy_steps` + `_step_to_event` shims make both
layouts surface as the same `TrajectoryEvent` stream from
`TrajectoryView.__iter__`, so this writer is layout-agnostic.

Writes `transcript.txt` (consolidated) + per-event files at
`<out_dir>/events/NNN_<kind>.txt`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cube_harness.core import (
    AgentErrorEvent,
    EvaluationEvent,
    LLMCallEvent,
    ToolCallEvent,
    TrajectoryEvent,
)
from cube_harness.storage import FileStorage

logger = logging.getLogger(__name__)


def _format_llm_call(idx: int, te: TrajectoryEvent) -> str:
    """Render one LLMCallEvent as a readable text block.

    LLM calls carry the prompt + assistant response + token usage. We
    surface the assistant's content, any thinking/reasoning trace, tool
    calls embedded in the response, and a one-line usage summary.
    """
    out = te.output
    assert isinstance(out, LLMCallEvent)
    lines = [f"### Event {idx:03d} LLM_CALL  (tag={(out.call.tag if out.call else '') or '?'})"]
    if out.error is not None:
        lines.append(f"ERROR: {out.error.error_type}: {out.error.exception_str}")
    if out.call is None:
        lines.append("(no LLMCall payload — legacy V1 episode)")
        return "\n".join(lines).rstrip() + "\n"

    msg = out.call.output
    content = getattr(msg, "content", None)
    if content:
        lines.append("RESPONSE:")
        lines.append(str(content))

    # Reasoning / thinking trace — model-dependent attributes.
    for attr in ("reasoning_content", "thinking_blocks"):
        val = getattr(msg, attr, None)
        if val:
            lines.append(f"{attr.upper()}:")
            lines.append(str(val))

    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", "?")
        args = getattr(getattr(tc, "function", None), "arguments", "")
        lines.append(f"TOOL_CALL {name}:\n{args}")

    usage = out.call.usage
    if usage is not None:
        lines.append(f"USAGE: prompt={usage.prompt_tokens} completion={usage.completion_tokens} cost=${usage.cost:.4f}")
    return "\n".join(lines).rstrip() + "\n"


def _format_tool_call(idx: int, te: TrajectoryEvent) -> str:
    """Render one ToolCallEvent as a readable text block.

    Combines what V1's split `_act` + `_obs` files surfaced: the action
    that was dispatched (from the new `action` field), the observation
    that came back, and any error.
    """
    out = te.output
    assert isinstance(out, ToolCallEvent)
    lines = [f"### Event {idx:03d} TOOL_CALL  (parent_event_id={out.parent_event_id[:8]}…)"]
    if out.action is not None:
        name = out.action.name
        try:
            args = json.dumps(out.action.arguments, default=str, indent=2)
        except Exception:
            args = str(out.action.arguments)
        lines.append(f"ACTION {name}:\n{args}")
    elif out.action_id:
        lines.append(f"ACTION (id only): {out.action_id}")

    if out.error is not None:
        lines.append(f"ERROR: {out.error.error_type}: {out.error.exception_str}")

    contents = list(getattr(out.obs, "contents", []) or [])
    if contents:
        lines.append("OBS:")
        for c in contents:
            data: Any = getattr(c, "data", c) if not isinstance(c, dict) else c.get("data", "")
            if isinstance(data, bytes):
                data = f"<binary {len(data)} bytes>"
            tool_call_id = getattr(c, "tool_call_id", None) if not isinstance(c, dict) else c.get("tool_call_id")
            if tool_call_id:
                lines.append(f"[tool_call_id={tool_call_id}]")
            lines.append(str(data))
    return "\n".join(lines).rstrip() + "\n"


def _format_eval(idx: int, te: TrajectoryEvent) -> str:
    out = te.output
    assert isinstance(out, EvaluationEvent)
    flavor = "TERMINAL" if out.is_terminal else "STEP"
    lines = [f"### Event {idx:03d} EVAL ({flavor})  reward={out.reward}"]
    if out.info:
        try:
            lines.append(json.dumps(out.info, default=str, indent=2))
        except Exception:
            lines.append(str(out.info))
    return "\n".join(lines).rstrip() + "\n"


def _format_agent_error(idx: int, te: TrajectoryEvent) -> str:
    out = te.output
    assert isinstance(out, AgentErrorEvent)
    return f"### Event {idx:03d} AGENT_ERROR\n{out.error.error_type}: {out.error.exception_str}\n"


def _format_event(idx: int, te: TrajectoryEvent) -> tuple[str, str]:
    """Dispatch one event to its formatter. Returns (kind, text)."""
    out = te.output
    if isinstance(out, LLMCallEvent):
        return "llm", _format_llm_call(idx, te)
    if isinstance(out, ToolCallEvent):
        return "tool_call", _format_tool_call(idx, te)
    if isinstance(out, EvaluationEvent):
        return "eval", _format_eval(idx, te)
    if isinstance(out, AgentErrorEvent):
        return "agent_error", _format_agent_error(idx, te)
    return "unknown", f"### Event {idx:03d} UNKNOWN ({type(out).__name__})\n"


def extract_transcript(episode_dir: Path, out_dir: Path) -> Path:
    """Decompress every event in an episode dir into readable .txt files.

    Layout-agnostic: routes through `FileStorage.load_episode(trajectory_id)`
    so legacy `steps/`-only episodes and new `events/`-layout episodes
    both produce the same text output. Writes one file per event into
    `<out_dir>/events/NNN_<kind>.txt` plus a consolidated `transcript.txt`.

    Returns `out_dir`.
    """
    # Episode-dir layout: `<output_dir>/episodes/<trajectory_id>/...`
    # FileStorage expects to be rooted at `<output_dir>`.
    trajectory_id = episode_dir.name
    output_dir = episode_dir.parent.parent
    storage = FileStorage(output_dir)
    view = storage.load_episode(trajectory_id)

    out_events = out_dir / "events"
    out_events.mkdir(parents=True, exist_ok=True)

    consolidated: list[str] = []
    for idx, te in enumerate(view):
        try:
            kind, text = _format_event(idx, te)
        except Exception as e:
            logger.warning("Failed to format event %d of %s: %s", idx, trajectory_id, e)
            continue
        (out_events / f"{idx:03d}_{kind}.txt").write_text(text)
        consolidated.append(text)

    (out_dir / "transcript.txt").write_text("\n".join(consolidated))
    return out_dir
