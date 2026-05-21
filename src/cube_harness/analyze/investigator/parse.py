"""Parsing and shape helpers shared between the investigator and its drivers.

This module holds:
- `TraceMode` and `INVESTIGATOR_ALLOWED_TOOLS` — protocol-level constants shared by
  drivers, the audit pass, and core.
- `extract_json_block` — pulls the investigator's final JSON object out of free text.
- `_summarise_tool_input` — one-line tool-call rendering for traces (private;
  only used by drivers internally).

Renamed from `sdk.py`: the original module was a thin shell around
`claude-agent-sdk`, but the SDK call has since moved to
`driver.ClaudeCodeSDKDriver`. Nothing here is SDK-shaped any more.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Tools the investigator needs: read transcript files, grep through cube/agent source,
# inspect screenshots if any. No write/edit — the investigator only produces a JSON answer.
INVESTIGATOR_ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep", "Bash")

TraceMode = Literal["actions", "full", "off"]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_block(text: str) -> dict[str, Any]:
    """Extract a JSON object from the investigator's final assistant message."""
    m = _JSON_FENCE_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        # Fall back to first {...} block at top level.
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in investigator output")
        candidate = text[start:]
    # Try strict parse first, then trim trailing chatter.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    end = candidate.rfind("}")
    if end == -1:
        raise ValueError("No closing brace in investigator output")
    return json.loads(candidate[: end + 1])


def _summarise_tool_input(name: str, raw_input: dict[str, Any]) -> str:
    """One-line summary of a Claude Code tool call argument set."""
    if not isinstance(raw_input, dict):
        return str(raw_input)[:80]
    if name == "Bash":
        cmd = str(raw_input.get("command", ""))
        return cmd if len(cmd) <= 100 else cmd[:97] + "..."
    if name == "Read":
        target = raw_input.get("file_path") or raw_input.get("path") or ""
        offset = raw_input.get("offset")
        limit = raw_input.get("limit")
        suffix = f" (offset={offset}, limit={limit})" if offset or limit else ""
        return f"{target}{suffix}"
    if name == "Grep":
        pattern = raw_input.get("pattern", "")
        path = raw_input.get("path") or raw_input.get("glob") or ""
        return f"{pattern!r} in {path}" if path else repr(pattern)
    if name == "Glob":
        return str(raw_input.get("pattern", ""))
    # Fallback: compact JSON, truncated.
    try:
        s = json.dumps(raw_input, default=str)
    except Exception:
        s = str(raw_input)
    return s if len(s) <= 100 else s[:97] + "..."


__all__ = [
    "extract_json_block",
    "_summarise_tool_input",
    "TraceMode",
    "INVESTIGATOR_ALLOWED_TOOLS",
]
