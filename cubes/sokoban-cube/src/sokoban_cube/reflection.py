"""Sokoban self-reflection content — the LaMer-paper reflection prompt for cross-episode rollouts.

cube-harness owns the generic reflection *mechanism* (``CrossEpisodeReflector`` /
``ReflectionProvider``); this module owns the Sokoban *content*. Faithful to the LaMer paper's Sokoban
Reflection Prompt (Appendix B.1): it replays the failed attempt — **initial board → actions taken →
final board → "task NOT completed"** — then has the model **reason step-by-step** and commit an
improved plan inside ``<remark></remark>`` tags; only the ``<remark>`` is forwarded as the lesson to
the next attempt (``conclusion_tag``). Mirrors math-tool-use's ``MathReflectionProvider`` (v3_lamer);
the board glyphs match this cube's renderer (``# @ $ . * +``), not the paper's X/O/P legend, so the
model reads the same boards it played on.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from litellm import Message

from cube_harness.agents.reflection import History, ReflectionProvider, ReflectionProviderConfig
from cube_harness.llm import Prompt

logger = logging.getLogger(__name__)

# System message for the reflection turn — environment + the "reflect on a past attempt" task framing
# (paper Appendix B.1), with THIS cube's glyphs so the replayed boards are read correctly.
_REFLECT_SYSTEM = """\
You are an expert agent operating in the Sokoban environment, a puzzle where you push boxes onto goal \
squares on a grid.

Board glyphs: # wall, @ you, $ box, . goal, * box-on-goal, + you-on-goal, (space) floor. You win when \
every box sits on a goal (every $ becomes *). You can only push a box (never pull), so a box shoved \
into a corner or flat against a wall away from its goal is permanently stuck — plan ahead.

# Your Task
You will be given the history of a past attempt at a level. Your job is to reflect on that sequence, \
identify any mistakes or inefficiencies, and devise a concise, improved plan starting from the \
original initial state."""

# User message — replays the attempt (initial board, actions, final board) and asks for
# step-by-step reasoning then an improved plan inside <remark></remark> (paper Appendix B.1).
_REFLECT_USER_TEMPLATE = """\
# Past Experience
The initial state of the level was:
{initial_board}

You took the following actions, in order:
{actions}

The final state was:
{final_board}

The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.
- Your response should first be step-by-step reasoning about the strategy and path you took to attempt \
the level. Identify where things went wrong or could be better (which box got stuck, which push was \
the mistake).
- Then devise a concise, new plan of action that accounts for your mistake, with reference to the \
specific moves you should have taken instead.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to \
guide the next trial."""

_LESSON_HEADER = "## Reflection and improved plan from your previous attempt(s) at THIS level:"

# A board row is glyphs-only and contains a wall; this cleanly pulls the rendered grid out of an
# observation/tool message (skipping instructions, legend, and the "Boxes on goals: X/N" status line).
_BOARD_LINE = re.compile(r"^[#@$.*+ ]+$")


def _msg_attr(msg: dict | Message, key: str) -> Any:
    """Read ``key`` from a history entry that may be a dict or a litellm ``Message``."""
    return msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)


def _extract_board(text: Any) -> str:
    """Pull the rendered board grid (glyph-only rows with a wall) out of a message's text."""
    if not isinstance(text, str):
        return ""
    rows = [ln for ln in text.splitlines() if "#" in ln and _BOARD_LINE.match(ln)]
    return "\n".join(rows)


def summarize_sokoban_history(history: History) -> tuple[str, str, str]:
    """(initial_board, actions, final_board) reconstructed from the agent's episode history.

    initial_board: grid from the first user observation. actions: every ``move(directions)`` tool call,
    in order, flattened to one comma-separated sequence. final_board: grid from the last tool result.
    """
    initial_board = ""
    final_board = ""
    actions: list[str] = []
    for msg in history:
        role = _msg_attr(msg, "role")
        content = _msg_attr(msg, "content")
        if role == "user" and not initial_board:
            initial_board = _extract_board(content)
        if role in ("tool", "function"):
            board = _extract_board(content)
            if board:
                final_board = board
        for tc in _msg_attr(msg, "tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) if fn else None
            if name != "move":
                continue
            raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (ValueError, TypeError):
                args = {}
            directions = args.get("directions") if isinstance(args, dict) else None
            if isinstance(directions, str) and directions.strip():
                actions.append(directions.strip())
    return (
        initial_board or "(initial board unavailable)",
        ", ".join(actions) or "(no moves were made)",
        final_board or "(final board unavailable)",
    )


class SokobanReflectionProvider:
    """Sokoban ``ReflectionProvider``: replay the failed attempt into the paper's reflection prompt,
    reason → ``<remark>`` plan, forward only the ``<remark>`` as the lesson."""

    def __init__(self, conclusion_tag: str = "remark") -> None:
        self._tag = conclusion_tag

    def build_reflection_prompt(
        self, history: History, *, reward: float, max_actions: int, requirements: str | None
    ) -> Prompt:
        _ = reward, max_actions, requirements  # self-diagnose from the replayed trajectory
        initial_board, actions, final_board = summarize_sokoban_history(history)
        user = _REFLECT_USER_TEMPLATE.format(initial_board=initial_board, actions=actions, final_board=final_board)
        return Prompt(
            messages=[
                {"role": "system", "content": _REFLECT_SYSTEM},
                {"role": "user", "content": user},
            ]
        )

    def format_lessons(self, reflections: list[str], *, max_actions: int) -> str:
        _ = max_actions
        entries = "\n\n".join(f"### After attempt {i + 1}\n{r}" for i, r in enumerate(reflections))
        return f"{_LESSON_HEADER}\n\n{entries}"

    def extract_lesson(self, raw_reflection: str) -> str:
        """Forward only the last ``<remark>…</remark>`` block (the committed plan); fall back to the
        full reflection if the model omitted the tag."""
        matches = re.findall(rf"<{self._tag}>(.*?)</{self._tag}>", raw_reflection, re.DOTALL)
        if not matches:
            logger.warning("reflection: no <%s> block in output; forwarding the full reflection", self._tag)
            return raw_reflection.strip()
        return matches[-1].strip()


class SokobanReflectionProviderConfig(ReflectionProviderConfig):
    """Config for :class:`SokobanReflectionProvider` (LaMer-paper Sokoban reflection). ``conclusion_tag``
    is the block forwarded to the next attempt (``remark``, per the paper)."""

    conclusion_tag: str = "remark"

    def make(self) -> ReflectionProvider:
        return SokobanReflectionProvider(self.conclusion_tag)
