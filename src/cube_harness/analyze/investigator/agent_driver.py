"""`AgentDriver` — coding-agent transport abstraction.

LiteLLM is a model gateway, not a coding-agent abstraction. To swap Claude SDK ↔
terminal `claude -p` ↔ `codex exec`, we need our own thin Protocol. Drivers are
call-time arguments to `investigate_episode` / `investigate_experiment` — orthogonal to what
the investigator is asked to do (the `InvestigatorRecipe`).

Two concrete drivers ship in this module:

- `ClaudeCodeSDKDriver` — wraps `claude-agent-sdk`. Needs an Anthropic API key;
  high parallelism (~8). Reports cost.
- `TerminalClaudeDriver` — subprocess-wraps the `claude -p` headless CLI. Works
  for subscription holders who don't have an API key; lower parallelism (~2).
  Does not report cost.

Both honour `LITELLM_PROXY_URL` / `LITELLM_PROXY_AUTH_TOKEN` by setting
`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` for the SDK / subprocess. The
proxy URL (without credentials) is recorded on `DriverResult` so investigations
remain auditable across deployments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from cube.core import TypedBaseModel
from pydantic import Field

from cube_harness.analyze.investigator.parse import INVESTIGATOR_ALLOWED_TOOLS, TraceMode, _summarise_tool_input

logger = logging.getLogger(__name__)


class ToolAction(TypedBaseModel):
    """One coding-agent tool call observed during a driver invocation."""

    tool: str
    input_summary: str
    raw_input: dict[str, Any] | None = None


class DriverResult(TypedBaseModel):
    """Everything the investigator needs back from a driver run.

    `cost_usd == 0.0` is sentinel-overloaded for "driver does not meter" — the
    terminal driver returns 0 because the headless CLI doesn't emit a cost in its
    JSON; check `litellm_proxy_url is None` to disambiguate from a free run.
    """

    output_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    actions: list[ToolAction] = Field(default_factory=list)
    litellm_proxy_url: str | None = None
    session_id: str | None = None


@runtime_checkable
class AgentDriver(Protocol):
    """Pluggable coding-agent transport.

    Implementations must be import-light — they may probe their dependency at
    `__init__` but should not perform network I/O until `run` is awaited.
    """

    name: str
    max_parallelism: int

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cwd: Path,
        additional_dirs: list[Path],
        model: str,
        allowed_tools: tuple[str, ...] = INVESTIGATOR_ALLOWED_TOOLS,
        permission_mode: Literal["bypassPermissions", "ask"] = "bypassPermissions",
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult: ...

    async def continue_session(
        self,
        *,
        session_id: str,
        follow_up_prompt: str,
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult: ...


# ---------------------------------------------------------------------------
# Helpers shared between drivers
# ---------------------------------------------------------------------------


def _proxy_env_overrides() -> tuple[dict[str, str], str | None]:
    """Map `LITELLM_PROXY_URL` to `ANTHROPIC_BASE_URL` (and matching auth token).

    Returns `(env_overrides, proxy_url_for_audit)`. When the proxy is not set,
    returns `({}, None)`. The auth-token mapping mirrors LiteLLM's docs: callers
    set `LITELLM_PROXY_AUTH_TOKEN` (or fall back to whichever Anthropic token is
    already in the env), and we forward it as `ANTHROPIC_AUTH_TOKEN`.

    The returned `proxy_url_for_audit` is the URL alone — never the token —
    because it ends up persisted in `DriverResult` and (eventually) in CSV reports.
    """
    proxy = os.environ.get("LITELLM_PROXY_URL")
    if not proxy:
        return {}, None
    overrides = {"ANTHROPIC_BASE_URL": proxy}
    token = os.environ.get("LITELLM_PROXY_AUTH_TOKEN")
    if token:
        overrides["ANTHROPIC_AUTH_TOKEN"] = token
    return overrides, proxy


def _summarise(name: str, raw_input: Any) -> str:
    """Wrap the existing helper for the new ToolAction shape."""
    if isinstance(raw_input, dict):
        return _summarise_tool_input(name, raw_input)
    return str(raw_input)[:140]


# ---------------------------------------------------------------------------
# ClaudeCodeSDKDriver
# ---------------------------------------------------------------------------


class ClaudeCodeSDKDriver:
    """Wraps the `claude-agent-sdk` Python package.

    The SDK manages its own subprocess pool, so `run` is a thin async generator
    consumer; we collect text/tool-use blocks and the final `ResultMessage` for
    usage and cost reporting.
    """

    name: str = "claude-code-sdk"
    max_parallelism: int = 8

    def __init__(self, *, max_parallelism: int | None = None) -> None:
        if max_parallelism is not None:
            self.max_parallelism = max_parallelism

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cwd: Path,
        additional_dirs: list[Path],
        model: str,
        allowed_tools: tuple[str, ...] = INVESTIGATOR_ALLOWED_TOOLS,
        permission_mode: Literal["bypassPermissions", "ask"] = "bypassPermissions",
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                ToolUseBlock,
                query,
            )
        except ImportError as e:
            raise RuntimeError("claude-agent-sdk not installed. Run: pip install 'cube-harness[investigator]'") from e

        env_overrides, proxy_url = _proxy_env_overrides()
        prior_env: dict[str, str | None] = {}
        for key, value in env_overrides.items():
            prior_env[key] = os.environ.get(key)
            os.environ[key] = value

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=list(allowed_tools),
            permission_mode=permission_mode,
            cwd=str(cwd),
            add_dirs=[str(p) for p in additional_dirs],
            model=model,
            include_partial_messages=False,
        )
        if verbose:
            logger.info("Running investigator with model %s and options: %r", model, options)

        final_text: list[str] = []
        collected_actions: list[ToolAction] = []
        prompt_tokens = 0
        completion_tokens = 0
        cost_usd = 0.0
        duration_ms = 0
        session_id: str | None = None
        start = time.time()

        def _emit(line: str) -> None:
            print(line, file=sys.stderr, flush=True)

        try:
            async for message in query(prompt=user_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_text.append(block.text)
                            if verbose and block.text.strip():
                                _emit(f"  · {block.text.strip().splitlines()[0][:140]}")
                        elif isinstance(block, ToolUseBlock):
                            summary = _summarise(block.name, block.input)
                            if verbose:
                                _emit(f"  > {block.name}({summary})")
                            if trace_mode in ("actions", "full"):
                                raw = block.input if (trace_mode == "full" and isinstance(block.input, dict)) else None
                                collected_actions.append(
                                    ToolAction(tool=block.name, input_summary=summary, raw_input=raw)
                                )
                elif isinstance(message, ResultMessage):
                    usage = getattr(message, "usage", None) or {}
                    prompt_tokens = (
                        int(usage.get("input_tokens", 0) or 0)
                        + int(usage.get("cache_read_input_tokens", 0) or 0)
                        + int(usage.get("cache_creation_input_tokens", 0) or 0)
                    )
                    completion_tokens = int(usage.get("output_tokens", 0) or 0)
                    cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
                    duration_ms = int(getattr(message, "duration_ms", 0) or 0)
                    session_id = getattr(message, "session_id", None)
        finally:
            for key, prev in prior_env.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

        duration_s = duration_ms / 1000.0 if duration_ms else (time.time() - start)
        return DriverResult(
            output_text="\n".join(final_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            duration_s=duration_s,
            actions=collected_actions if trace_mode != "off" else [],
            litellm_proxy_url=proxy_url,
            session_id=session_id,
        )

    async def continue_session(
        self,
        *,
        session_id: str,
        follow_up_prompt: str,
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult:
        # The Python SDK does not currently expose a stable "resume by session id"
        # entry point in the version we depend on. Caller falls back to a fresh
        # `run` with prior context serialised in.
        raise NotImplementedError(
            "ClaudeCodeSDKDriver.continue_session is not supported by the current SDK; "
            "audit pass should fall back to a fresh run with prior finding in the prompt."
        )


# ---------------------------------------------------------------------------
# TerminalClaudeDriver
# ---------------------------------------------------------------------------


class TerminalClaudeDriver:
    """Subprocess wrapper around the `claude -p` headless CLI.

    Use case: subscription holders who can run `claude` locally but don't have an
    Anthropic API key. The CLI's JSON output mode emits an envelope we parse for
    the assistant's final text and (where available) tool-call traces.

    Cost reporting is left at zero — the headless CLI does not surface a per-run
    cost in its JSON envelope. Token counts are populated when the CLI provides
    them. Parallelism is conservative (2) because each invocation spawns a full
    Node process.
    """

    name: str = "claude-terminal"
    max_parallelism: int = 2

    def __init__(
        self,
        *,
        executable: str = "claude",
        max_parallelism: int | None = None,
    ) -> None:
        self.executable = executable
        if max_parallelism is not None:
            self.max_parallelism = max_parallelism

    def _build_args(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        cwd: Path,
        additional_dirs: list[Path],
        model: str,
        allowed_tools: tuple[str, ...],
        permission_mode: Literal["bypassPermissions", "ask"],
    ) -> list[str]:
        args: list[str] = [
            self.executable,
            "-p",
            user_prompt,
            "--append-system-prompt",
            system_prompt,
            "--allowedTools",
            ",".join(allowed_tools),
            "--output-format",
            "json",
            "--model",
            model,
        ]
        for d in additional_dirs:
            args.extend(["--add-dir", str(d)])
        if permission_mode == "bypassPermissions":
            args.append("--dangerously-skip-permissions")
        return args

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        cwd: Path,
        additional_dirs: list[Path],
        model: str,
        allowed_tools: tuple[str, ...] = INVESTIGATOR_ALLOWED_TOOLS,
        permission_mode: Literal["bypassPermissions", "ask"] = "bypassPermissions",
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult:
        env_overrides, proxy_url = _proxy_env_overrides()
        env = {**os.environ, **env_overrides}

        args = self._build_args(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            cwd=cwd,
            additional_dirs=additional_dirs,
            model=model,
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
        )
        if verbose:
            logger.info("TerminalClaudeDriver.exec: %s", " ".join(args[:8]) + " ...")

        start = time.time()
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        duration_s = time.time() - start

        if proc.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"claude -p exited with code {proc.returncode}: {stderr.strip()[-400:] or '<no stderr>'}"
            )

        return self._parse_envelope(
            stdout_bytes,
            duration_s=duration_s,
            proxy_url=proxy_url,
            trace_mode=trace_mode,
        )

    def _parse_envelope(
        self,
        stdout_bytes: bytes,
        *,
        duration_s: float,
        proxy_url: str | None,
        trace_mode: TraceMode,
    ) -> DriverResult:
        """Extract the relevant fields from the `claude -p --output-format json` envelope."""
        text = stdout_bytes.decode("utf-8", errors="replace").strip()
        try:
            envelope: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude -p emitted non-JSON: {text[:400]!r}") from e

        # Envelope shape varies across CLI versions; we look up by best-effort.
        output_text = (
            envelope.get("result")
            or envelope.get("output")
            or envelope.get("text")
            or envelope.get("message", {}).get("content", "")
            or ""
        )
        if isinstance(output_text, list):
            # Some versions emit content as a list of blocks.
            parts = []
            for block in output_text:
                if isinstance(block, dict):
                    parts.append(block.get("text") or block.get("content") or "")
                else:
                    parts.append(str(block))
            output_text = "\n".join(p for p in parts if p)

        usage = envelope.get("usage") or {}
        prompt_tokens = int(
            (usage.get("input_tokens", 0) or 0)
            + (usage.get("cache_read_input_tokens", 0) or 0)
            + (usage.get("cache_creation_input_tokens", 0) or 0)
        )
        completion_tokens = int(usage.get("output_tokens", 0) or 0)

        actions: list[ToolAction] = []
        if trace_mode != "off":
            for action in envelope.get("tool_uses", []) or envelope.get("actions", []) or []:
                if not isinstance(action, dict):
                    continue
                tool_name = action.get("name") or action.get("tool") or "?"
                raw = action.get("input") or action.get("arguments") or {}
                summary = _summarise(tool_name, raw)
                actions.append(
                    ToolAction(
                        tool=tool_name,
                        input_summary=summary,
                        raw_input=raw if (trace_mode == "full" and isinstance(raw, dict)) else None,
                    )
                )

        return DriverResult(
            output_text=str(output_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,  # claude -p does not surface cost
            duration_s=duration_s,
            actions=actions,
            litellm_proxy_url=proxy_url,
            session_id=envelope.get("session_id"),
        )

    async def continue_session(
        self,
        *,
        session_id: str,
        follow_up_prompt: str,
        verbose: bool = False,
        trace_mode: TraceMode = "actions",
    ) -> DriverResult:
        # The headless CLI does support `--resume <session_id>`, but the audit
        # pass works just as well via a fresh run with the prior finding
        # serialised in. Keep this path simple — if a future version needs true
        # session continuation we can wire it through `--resume`.
        raise NotImplementedError(
            "TerminalClaudeDriver.continue_session is not implemented; "
            "audit pass falls back to a fresh run with prior finding in the prompt."
        )


__all__ = [
    "AgentDriver",
    "DriverResult",
    "ToolAction",
    "TraceMode",
    "INVESTIGATOR_ALLOWED_TOOLS",
    "ClaudeCodeSDKDriver",
    "TerminalClaudeDriver",
]
