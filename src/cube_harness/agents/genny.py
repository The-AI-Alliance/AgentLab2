"""Genny agent — cache-friendly context management.

Two operating modes, both with a stable cacheable prefix:

Mode A — growing raw history (enable_summarize=False):
    system_prompt          (static)
    goal                   (static)
    hints / clarification  (static)
    [obs_0, asst_0, ...]   (completed rounds, grow by one pair each step)
    latest_obs             (the new observation, appended at context-build time)
    react_prompt           (static)

Mode B — rolling summaries (enable_summarize=True):
    system_prompt          (static)
    goal                   (static)
    hints / clarification  (static)
    asst: summary_1        (one message per step — NOT bundled)
    asst: summary_2
    ...
    asst: summary_k        ← rolling cache breakpoint lands here
    latest_obs             (shown to both sum and act passes)
    [asst: summary_{k+1}]  (act pass only, after summarize generates it)
    act_prompt / react_prompt

Summaries as separate messages (not a single bundled block) lets Anthropic's
prefix cache extend cleanly across steps: each step's cache ending at summary_k
is a valid prefix of the next step, which starts the same way and appends one more.
"""

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

from cube.core import Action, ActionSchema, Observation, ValidatedConfig
from cube.task import STOP_ACTION
from litellm import Message
from pydantic import Field
from termcolor import colored

from cube_harness.agent import Agent, AgentConfig
from cube_harness.core import AgentOutput
from cube_harness.llm import LLM, LLMCall, LLMConfig, Prompt, get_reasoning

logger = logging.getLogger(__name__)


class Profiler:
    """Records named wall-clock spans; call as a context manager to record each span."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, float]] = {}

    @contextmanager
    def __call__(self, name: str) -> Generator[None, None, None]:
        t_start = time.time()
        yield
        self._data[name] = (t_start, time.time())

    @property
    def data(self) -> dict[str, tuple[float, float]]:
        return self._data


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are an expert AI agent. Understand the goal, take targeted actions, and reason clearly about progress.
Verify that each action had the intended effect before proceeding. Be concise and focused."""

_DEFAULT_REACT_PROMPT = """\
Review the latest observation and produce the next action.
Think step by step:
1. What does the observation show?
2. Did the last action have the intended effect? If the page state is unchanged or the action failed, do NOT repeat it — try a different element, method, or approach.
3. What is the best next action?
Then call the appropriate function."""

_DEFAULT_ACT_PROMPT = """\
Based on the reasoning above, call the appropriate function to perform the next action."""

_DEFAULT_SUMMARIZE_VERBOSE_PROMPT = """\
Summarize the latest observation concisely. Include:
- What was observed (key changes, current state, errors)
- Progress toward the goal

Then add a '## Key Facts' section with durable facts worth preserving across compactions.

Respond with text only — do not call any tools or functions."""

_DEFAULT_SUMMARIZE_COT_PROMPT = """\
In 2-3 sentences, reason about the latest observation: what happened, what it means for the goal, and what to do next.

Respond with text only — do not call any tools or functions."""

_DEFAULT_COMPACT_PROMPT = """\
Summarize the conversation history above. Include:
- Key actions taken and their results
- Important findings about the codebase
- Current state and what has been accomplished so far
Be concise but preserve all information needed to continue the task.
Respond with text only — do not call any tools."""


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def _encode_tools(tools: list[ActionSchema]) -> list[dict]:
    return [t.as_dict() for t in tools]


def _decode_actions(response: "Message") -> "list[Action]":
    actions = []
    for tc in getattr(response, "tool_calls", None) or []:
        args = tc.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        if tc.function.name:
            actions.append(Action(id=tc.id, name=tc.function.name, arguments=args))
    return actions


def _format_action_list(actions: "list[Action]") -> str:
    """Format a list of actions as a compact text string."""
    parts = [f"{a.name}({', '.join(f'{k}={v!r}' for k, v in a.arguments.items())})" for a in actions]
    return ", ".join(parts) if parts else "no action"


def _truncate_message(msg: dict, max_chars: int) -> dict:
    content = msg.get("content", "")
    if isinstance(content, str) and len(content) > max_chars:
        return {**msg, "content": content[:max_chars] + "… [truncated]"}
    return msg


# ---------------------------------------------------------------------------
# Budget config
# ---------------------------------------------------------------------------


class BudgetConfig(ValidatedConfig):
    """Episode budget limits and periodic status display.

    Single entry point: check(cost, tokens, step) -> (status_msg | None, busted).
    - busted=True when any configured limit is exceeded → caller should STOP.
    - status_msg is a human-readable summary injected into the prompt every
      display_every_k steps; None when it is not a display step or nothing is configured.

    Format: "cumulative episode budget: 34/150 steps, 35% token usage."
    Token usage % = max(cost/cost_limit, tokens/token_limit) across whichever limits are set.
    """

    max_actions: int | None = None
    cost_limit: float | None = None
    token_limit: int | None = None
    display_every_k: int = 5

    def check(self, cost: float, tokens: int, step: int) -> tuple[str | None, bool]:
        """Return (status_message_or_None, is_over_budget)."""
        busted = (
            (self.max_actions is not None and step >= self.max_actions)
            or (self.cost_limit is not None and cost >= self.cost_limit)
            or (self.token_limit is not None and tokens >= self.token_limit)
        )
        if busted:
            return None, True
        msg: str | None = None
        if step > 0 and step % self.display_every_k == 0:
            parts: list[str] = []
            if self.max_actions is not None:
                parts.append(f"{step}/{self.max_actions} steps")
            if self.cost_limit is not None or self.token_limit is not None:
                pct = 0.0
                if self.cost_limit is not None and self.cost_limit > 0:
                    pct = max(pct, cost / self.cost_limit)
                if self.token_limit is not None and self.token_limit > 0:
                    pct = max(pct, tokens / self.token_limit)
                parts.append(f"{pct * 100:.1f}% token usage")
            if parts:
                msg = "cumulative episode budget: " + ", ".join(parts) + "."
        return msg, False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GennyConfig(AgentConfig):
    # Core
    llm_config: LLMConfig
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    # react_prompt: reason-then-act, used when enable_summarize=False and flat_history=False
    react_prompt: str = _DEFAULT_REACT_PROMPT
    # act_prompt: action-only, used when enable_summarize=True and flat_history=False
    act_prompt: str = _DEFAULT_ACT_PROMPT
    # step_prompt: trailing user message appended each act step when flat_history=True.
    # Empty string = no trailing message (mini-swe-agent style).
    step_prompt: str = ""

    # goal_template: template applied to the first observation (the task/problem statement).
    # Use "{{task}}" as the placeholder for the raw observation text.
    # Empty string = use raw observation text unchanged (default).
    # Useful for wrapping the issue in <pr_description> + <instructions> blocks (mini-swe-agent style).
    goal_template: str = ""

    # Flat history mode: True = linear conversation with no injected summaries/headers,
    # equivalent to mini-swe-agent prompt structure. Summaries are still accumulated
    # internally (for logging/XRay) but not injected into the prompt.
    flat_history: bool = False

    # Summarize pass
    enable_summarize: bool = False  # False = raw history mode; True = rolling summaries mode
    summarize_llm_config: LLMConfig | None = None  # None = reuse llm_config
    # Instruction sent to the summarize LLM. Swap to _DEFAULT_SUMMARIZE_COT_PROMPT for a
    # lighter CoT-style summary instead of the default verbose + Key Facts format.
    summarize_prompt: str = _DEFAULT_SUMMARIZE_VERBOSE_PROMPT

    # General hint injected after the goal in every step's context.
    # Use this when one hint applies to a whole task subset (one config per subset).
    hint: str = ""

    # Per-task hints: task_id -> hint text. Takes precedence over `hint` when a task_id match is found.
    # These are general or task-specific hints that help the LLM work better.
    task_hints: dict[str, str] = Field(default_factory=dict)

    # Per-task precision: task_id -> text that clarifies the goal when the task description
    # is under-defined (e.g. expected answer format, submission method). Injected as part of
    # the goal — not as a separate hint section.
    task_clarification: dict[str, str] = Field(default_factory=dict)

    # Observation format for tool results (role="tool" messages).
    # "raw"        = send content unchanged (default).
    # "output_tag" = wrap content in <output>...</output>, matching mini-swe-agent format.
    obs_format: str = "raw"

    # Context compaction — triggered when accumulated history exceeds this char threshold.
    # For flat_history=True: compacts self.history into a summary injected into the system message.
    # For enable_summarize=True (flat_history=False): compacts self.summaries into one entry.
    # None = disabled (default).
    compact_threshold_chars: int | None = None
    compact_prompt: str = _DEFAULT_COMPACT_PROMPT

    # Misc
    max_obs_chars: int | None = None  # None = no truncation
    # Episode budget: step/cost/token limits + periodic status display.
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    # Retry budget when the model returns no tool calls. On each retry the empty response
    # and a correction user message are appended; if still no tool calls after all retries,
    # a STOP action is returned. 0 = no retry (preserves current behavior).
    max_format_errors: int = 0

    @property
    def agent_name(self) -> str:
        name = f"Genny-{self.llm_config.model_name}".replace("/", "_")
        if self.summarize_llm_config and self.summarize_llm_config.model_name != self.llm_config.model_name:
            name += f"+{self.summarize_llm_config.model_name}".replace("/", "_")
        return name

    def make(self, action_set: list[ActionSchema] | None = None, task_id: str | None = None, **kwargs) -> "Genny":
        return Genny(config=self, action_schemas=action_set or [], task_id=task_id)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Genny(Agent):
    """ReAct-style agent with cache-friendly context management and mini-swe-agent compatibility.

    Mode A (enable_summarize=False, flat_history=False): raw history grows by one (obs, asst)
    pair per step. Completed history is a stable cacheable prefix.

    Mode B (enable_summarize=True, flat_history=False): a separate summarize LLM call produces
    a per-step summary stored as its own assistant message. Summaries accumulate as individual
    messages so each step's cache extends the previous step's.

    Flat mode (flat_history=True): linear prompt with no injected scaffolding — equivalent to
    mini-swe-agent. Summaries (if enable_summarize=True) are computed for logging/XRay but
    never injected into the prompt. step_prompt="" omits the trailing user message entirely.
    """

    name: str = "genny"
    description: str = "Genny — cache-friendly context management with flat/summarize/raw history modes."
    input_content_types: list[str] = ["image/png", "image/jpeg", "text/plain", "application/json"]
    output_content_types: list[str] = ["application/json"]

    def __init__(self, config: GennyConfig, action_schemas: list[ActionSchema], task_id: str | None = None):
        self.config = config
        self.task_id = task_id
        if task_id is None and (config.task_hints or config.task_clarification):
            logger.debug(
                "task_id is None — %d task_hints and %d task_clarifications not applied",
                len(config.task_hints),
                len(config.task_clarification),
            )
        # task_hints takes precedence over the general hint; falls back to hint if no match.
        self._task_hint: str = config.task_hints.get(task_id, config.hint) if task_id else config.hint
        # task_clarification is injected as part of the goal, not as a hint.
        self._task_clarification: str = config.task_clarification.get(task_id, "") if task_id else ""
        self.llm: LLM = config.llm_config.make()
        # Summarize LLM uses the same config as the act LLM (including tool_choice) so the
        # full request — messages, tools, and parameters — is identical between the two passes
        # → prompt-cache hit on the shared prefix. tool_choice is intentionally NOT overridden
        # to "none" because Azure/OpenAI include it in the cache key.
        self._summarize_llm_config = config.summarize_llm_config or config.llm_config
        self.summarize_llm: LLM = self._summarize_llm_config.make()
        self.token_counter = config.llm_config.make_counter()
        self.action_schemas: list[ActionSchema] = action_schemas
        self.goal: list[dict] = []
        self.summaries: list[str] = []  # Mode B: one summary per step (raw, no action suffix)
        self.summary_actions: list[str] = []  # Mode B: action taken per step, separate message for cache stability
        self.history: list[list[dict | Message]] = []  # Mode A / flat: completed (obs, asst) pairs
        self._latest_obs: list[dict | Message] = []  # current step's obs, not yet in history
        self._actions_cnt: int = 0
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._compacted_summary: str = ""  # injected into system message after compaction

    def step(self, obs: Observation) -> AgentOutput:
        budget_msg, busted = self.config.budget.check(self._total_cost, self._total_tokens, self._actions_cnt)
        if busted:
            logger.info(
                "Budget limit reached (step=%d cost=$%.4f tokens=%d), issuing STOP.",
                self._actions_cnt,
                self._total_cost,
                self._total_tokens,
            )
            return AgentOutput(actions=[Action(name=STOP_ACTION.name, arguments={})])

        profiler = Profiler()

        with profiler("context"):
            obs_messages = self._obs_to_messages(obs)
            self._ingest_obs(obs_messages)

        compact_call: LLMCall | None = None
        with profiler("compact"):
            compact_call = self._compact_history()

        thoughts: str | None = None
        sum_call: LLMCall | None = None
        if self.config.enable_summarize:
            with profiler("summarize"):
                summary, sum_call = self._summarize_past()
            thoughts = summary
            self.summaries.append(summary)

        with profiler("act"):
            response, act_calls = self._act(budget_msg)
        actions = _decode_actions(response)

        # Format error exhaustion: _act() retried max_format_errors times but still no tool calls.
        if not actions and self.config.max_format_errors > 0:
            logger.warning("Format error retries exhausted — no tool calls returned. Issuing STOP.")
            actions = [Action(name=STOP_ACTION.name, arguments={})]

        if self.config.enable_summarize:
            self.summary_actions.append(_format_action_list(actions))
            if self.config.flat_history:
                # Flat mode: also commit obs+asst so _build_base_prompt renders the flat conversation.
                if self._latest_obs:
                    self.history.append(self._latest_obs)
                self.history.append([response])
        else:
            # Prefer the model's reasoning trace when available; otherwise fall back
            # to the response content so non-reasoning agents still surface their
            # inline ReAct thinking in the trajectory's `thoughts` panel.
            thoughts = get_reasoning(response) or response.content or None
            if self._latest_obs:
                self.history.append(self._latest_obs)
            self.history.append([response])

        llm_calls: list[LLMCall] = (
            act_calls
            + ([sum_call] if sum_call is not None else [])
            + ([compact_call] if compact_call is not None else [])
        )
        for call in llm_calls:
            self._total_cost += call.usage.cost
            self._total_tokens += call.usage.prompt_tokens + call.usage.completion_tokens
        self._actions_cnt += 1
        return AgentOutput(
            actions=actions,
            llm_calls=llm_calls,
            profiling=profiler.data,
            thoughts=thoughts or None,
        )

    def _obs_to_messages(self, obs: Observation) -> list[dict | Message]:
        messages = cast(list[dict | Message], obs.to_llm_messages())
        if self.config.obs_format == "output_tag":
            wrapped = []
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), str):
                    m = {**m, "content": f"<output>\n{m['content']}\n</output>"}
                wrapped.append(m)
            messages = cast(list[dict | Message], wrapped)
        if self.config.max_obs_chars is not None:
            messages = cast(list[dict | Message], [_truncate_message(m, self.config.max_obs_chars) for m in messages])
        return messages

    def _ingest_obs(self, obs_messages: list[dict | Message]) -> None:
        """On step 0 extract goal; on all steps park the obs in _latest_obs."""
        if not self.goal:
            first = obs_messages[0]
            if self.config.goal_template and "{{task}}" in self.config.goal_template:
                raw = first.get("content", "") if isinstance(first, dict) else str(first)
                first = {
                    **(first if isinstance(first, dict) else {}),
                    "content": self.config.goal_template.replace("{{task}}", raw),
                }
            self.goal = [first]
            self._latest_obs = list(obs_messages[1:])
        else:
            self._latest_obs = list(obs_messages)

    def _build_base_prompt(self, exclude_last_summary: bool = False) -> list[dict | Message]:
        """Build the stable prompt prefix shared by summarize and act passes.

        Mode A: system + goal + hints + completed history (obs+asst pairs).
        Mode B: system + goal + hints + summaries as individual assistant messages.

        Latest obs is NOT included here — callers append it so the prefix up to the
        last summary/action remains byte-identical across both passes and across steps,
        enabling Anthropic's longest-prefix cache matching.
        """
        system_content = self.config.system_prompt
        if self._compacted_summary:
            system_content += f"\n\n## Summary of earlier work\n{self._compacted_summary}"
        messages: list[dict | Message] = [{"role": "system", "content": system_content}]
        messages.extend(self.goal)
        if self._task_clarification:
            messages.append({"role": "user", "content": f"## Additional task details\n\n{self._task_clarification}"})
            messages.append({"role": "assistant", "content": "Understood."})
        if self._task_hint:
            messages.append({"role": "user", "content": f"## Task Hint\n\n{self._task_hint}"})
            messages.append({"role": "assistant", "content": "Understood, I'll keep this in mind."})
        if self.config.enable_summarize and not self.config.flat_history:
            summaries = self.summaries[:-1] if (exclude_last_summary and self.summaries) else self.summaries
            for i, s in enumerate(summaries):
                messages.append({"role": "assistant", "content": s})
                # Action for step i lives in a separate user message so the summary bytes
                # stay unchanged across steps → Anthropic prefix cache extends cleanly.
                if i < len(self.summary_actions):
                    messages.append({"role": "user", "content": self.summary_actions[i]})
        else:
            for group in self.history:
                messages.extend(group)
        return messages

    def _summarize_past(self) -> tuple[str, LLMCall]:
        """Summarize the latest obs. Prompt: base_prefix + latest_obs + summarize_instruction.

        The base_prefix (system, goal, hints, prior summaries) is byte-for-byte identical
        to the act pass prefix → within-step cache hit between summarize and act.
        """
        messages = self._build_base_prompt()
        messages.extend(self._latest_obs)
        messages.append({"role": "user", "content": self.config.summarize_prompt})
        api_tools = _encode_tools(self.action_schemas)
        prompt = Prompt(messages=messages, tools=api_tools)
        response = self.summarize_llm(prompt)
        llm_call = LLMCall(
            tag="summary",
            llm_config=self._summarize_llm_config,
            prompt=prompt,
            output=response.message,
            usage=response.usage,
        )
        return response.message.content or "", llm_call

    def _history_chars(self) -> int:
        """Estimate total chars in accumulated history (or summaries for Mode B)."""
        total = 0
        if self.config.enable_summarize and not self.config.flat_history:
            for s in self.summaries:
                total += len(s)
            for a in self.summary_actions:
                total += len(a)
        else:
            for group in self.history:
                for msg in group:
                    c = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "") or ""
                    if isinstance(c, str):
                        total += len(c)
        return total

    def _compact_history(self) -> "LLMCall | None":
        """Trigger compaction if accumulated history exceeds compact_threshold_chars."""
        if self.config.compact_threshold_chars is None:
            return None
        chars = self._history_chars()
        if chars < self.config.compact_threshold_chars:
            return None
        logger.info(f"Compaction triggered: {chars} chars > threshold {self.config.compact_threshold_chars}")
        if self.config.enable_summarize and not self.config.flat_history:
            return self._compact_summaries()
        return self._compact_flat_history()

    def _compact_flat_history(self) -> "LLMCall | None":
        """Compact flat history: summarise history[:-3], keep last 3 groups for API validity.

        Keeping history[-3:] = [asst_{N-1}, obs_N, asst_N] ensures that _latest_obs
        (tool results for asst_N's calls) remains properly preceded by a tool_use.
        """
        keep = 3
        if len(self.history) <= keep:
            return None
        messages: list[dict | Message] = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.goal)
        for group in self.history[:-keep]:
            messages.extend(group)
        messages.append({"role": "user", "content": self.config.compact_prompt})
        prompt = Prompt(messages=messages, tools=[])
        response = self.llm(prompt)
        summary = response.message.content or get_reasoning(response.message) or ""
        llm_call = LLMCall(
            tag="compact",
            llm_config=self.config.llm_config,
            prompt=prompt,
            output=response.message,
            usage=response.usage,
        )
        self._compacted_summary = summary
        self.history = self.history[-keep:]
        logger.info(f"Flat history compacted: {len(self.history)} groups remain ({self._history_chars()} chars)")
        return llm_call

    def _compact_summaries(self) -> "LLMCall | None":
        """Compact Mode B summaries list into a single consolidated summary."""
        if len(self.summaries) < 2:
            return None
        steps_text = "\n\n".join(
            f"Step {i + 1}:\n{s}" + (f"\nAction: {self.summary_actions[i]}" if i < len(self.summary_actions) else "")
            for i, s in enumerate(self.summaries)
        )
        messages: list[dict | Message] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": steps_text + "\n\n" + self.config.compact_prompt},
        ]
        prompt = Prompt(messages=messages, tools=[])
        response = self.llm(prompt)
        summary = response.message.content or get_reasoning(response.message) or ""
        llm_call = LLMCall(
            tag="compact",
            llm_config=self.config.llm_config,
            prompt=prompt,
            output=response.message,
            usage=response.usage,
        )
        self.summaries = [summary]
        self.summary_actions = []
        logger.info(f"Summaries compacted to 1 entry ({self._history_chars()} chars)")
        return llm_call

    def _act(self, budget_msg: str | None = None) -> tuple[Message, list[LLMCall]]:
        """Build context, encode tools, call act LLM; retry up to max_format_errors times on no tool calls."""
        messages = self._choose_context(budget_msg)
        api_tools = _encode_tools(self.action_schemas)
        prompt = Prompt(messages=messages, tools=api_tools)
        logger.info(f"Act pass — estimated prompt tokens: {self.token_counter(messages=messages)}")
        try:
            response = self.llm(prompt)
        except Exception as e:
            logger.exception(colored(f"LLM error in act pass: {e}", "red"))
            raise
        logger.info(
            f"LLM usage — prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens}, cost: ${response.usage.cost:.4f}"
        )
        llm_calls = [
            LLMCall(
                tag="act",
                llm_config=self.config.llm_config,
                prompt=prompt,
                output=response.message,
                usage=response.usage,
            )
        ]
        for attempt in range(self.config.max_format_errors):
            if response.message.tool_calls:
                break
            logger.warning(
                f"No tool calls in response (attempt {attempt + 1}/{self.config.max_format_errors}), retrying."
            )
            messages = list(messages) + [
                response.message,
                {"role": "user", "content": "No tool calls found. Every response MUST include at least one tool call."},
            ]
            prompt = Prompt(messages=messages, tools=api_tools)
            response = self.llm(prompt)
            llm_calls.append(
                LLMCall(
                    tag="act",
                    llm_config=self.config.llm_config,
                    prompt=prompt,
                    output=response.message,
                    usage=response.usage,
                )
            )
        return response.message, llm_calls

    def _choose_context(self, budget_msg: str | None = None) -> list[dict | Message]:
        """Build the act-pass prompt.

        Flat mode (flat_history=True): base_prefix (flat history) + latest_obs.
          step_prompt="" skips the trailing user message entirely — the model acts on
          the bare tool result, matching mini-swe-agent behavior.

        Mode B (enable_summarize=True, flat_history=False):
          base_prefix(exclude_last) + new_summary + latest_obs + act_prompt.
          Summaries are contiguous before the obs so the cache prefix ending at
          new_summary is a valid prefix of the next step's sum call.

        Mode A (enable_summarize=False, flat_history=False):
          base_prefix (completed history) + latest_obs + react_prompt.
        """
        if self.config.flat_history:
            messages = self._build_base_prompt()
            messages.extend(self._latest_obs)
            final_prompt = self.config.step_prompt
        elif self.config.enable_summarize:
            messages = self._build_base_prompt(exclude_last_summary=True)
            if self.summaries:
                messages.append({"role": "assistant", "content": self.summaries[-1]})
            messages.extend(self._latest_obs)
            final_prompt = self.config.act_prompt
        else:
            messages = self._build_base_prompt()
            messages.extend(self._latest_obs)
            final_prompt = self.config.react_prompt
        if budget_msg:
            final_prompt = f"{budget_msg}\n\n{final_prompt}" if final_prompt else budget_msg
        if final_prompt:
            messages.append({"role": "user", "content": final_prompt})
        return messages
