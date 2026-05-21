"""Step decoding: msgpack.zst → readable transcript text files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import msgpack
import zstandard

logger = logging.getLogger(__name__)


def _decompress(path: Path) -> dict[str, Any]:
    """Read and decompress a single step file (.msgpack.zst) into a plain dict."""
    with open(path, "rb") as f:
        data = f.read()
    dctx = zstandard.ZstdDecompressor()
    return msgpack.unpackb(dctx.decompress(data), raw=False)


def _format_obs(step_idx: int, raw: dict[str, Any]) -> str:
    """Render one observation step as a readable text block."""
    output = raw.get("output", raw)
    obs = output.get("obs", output) if isinstance(output, dict) else {}
    contents = obs.get("contents", []) if isinstance(obs, dict) else []
    reward = obs.get("reward") if isinstance(obs, dict) else None
    done = obs.get("done") if isinstance(obs, dict) else None

    lines = [f"### Step {step_idx:03d} OBS"]
    if reward is not None:
        lines.append(f"reward={reward}  done={done}")
    for c in contents:
        if not isinstance(c, dict):
            lines.append(str(c))
            continue
        tool_call_id = c.get("tool_call_id")
        data = c.get("data", "")
        if isinstance(data, bytes):
            data = f"<binary {len(data)} bytes>"
        if tool_call_id:
            lines.append(f"[tool_call_id={tool_call_id}]")
        lines.append(str(data))
    return "\n".join(lines).rstrip() + "\n"


def _format_act(step_idx: int, raw: dict[str, Any]) -> str:
    """Render one action step as a readable text block."""
    output = raw.get("output", raw)
    actions = output.get("actions", []) if isinstance(output, dict) else []
    llm_calls = output.get("llm_calls", []) if isinstance(output, dict) else []
    error = output.get("error") if isinstance(output, dict) else None

    lines = [f"### Step {step_idx:03d} ACT"]
    for call in llm_calls:
        thinking = call.get("thinking") if isinstance(call, dict) else None
        if thinking:
            lines.append("THINKING:")
            lines.append(str(thinking))
    for action in actions:
        if not isinstance(action, dict):
            lines.append(f"ACTION: {action}")
            continue
        name = action.get("name", "?")
        args = action.get("arguments", {})
        try:
            args_repr = json.dumps(args, default=str, indent=2)
        except Exception:
            args_repr = str(args)
        lines.append(f"ACTION {name}:\n{args_repr}")
    if error:
        lines.append(f"ERROR: {error}")
    return "\n".join(lines).rstrip() + "\n"


def extract_transcript(episode_dir: Path, out_dir: Path) -> Path:
    """Decompress every step file in `<episode_dir>/steps/` into readable .txt files.

    Writes one file per step into `<out_dir>/steps/NNN_(obs|act).txt` plus a
    consolidated `transcript.txt`. Returns `out_dir`.
    """
    steps_dir = episode_dir / "steps"
    if not steps_dir.exists():
        raise FileNotFoundError(f"No steps/ directory in {episode_dir}")

    out_steps = out_dir / "steps"
    out_steps.mkdir(parents=True, exist_ok=True)

    consolidated: list[str] = []
    for step_file in sorted(steps_dir.iterdir()):
        if not step_file.name.endswith(".msgpack.zst"):
            continue
        try:
            step_idx = int(step_file.name[:3])
        except ValueError:
            continue
        try:
            raw = _decompress(step_file)
        except Exception as e:
            logger.warning("Failed to decompress %s: %s", step_file, e)
            continue
        if "_obs" in step_file.name:
            text = _format_obs(step_idx, raw)
            (out_steps / f"{step_idx:03d}_obs.txt").write_text(text)
        elif "_act" in step_file.name:
            text = _format_act(step_idx, raw)
            (out_steps / f"{step_idx:03d}_act.txt").write_text(text)
        else:
            continue
        consolidated.append(text)

    (out_dir / "transcript.txt").write_text("\n".join(consolidated))
    return out_dir
