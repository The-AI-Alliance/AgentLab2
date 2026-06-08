"""Cross-episode self-reflection mechanism for LaMer-style multi-episode rollouts.

The reflection *mechanism* is domain-agnostic and lives here; the domain-specific *content* — what a
failed attempt looks like in the agent's history, how to diagnose it, and the prompt wording — is
supplied by a :class:`ReflectionProvider`, built from a serializable :class:`ReflectionProviderConfig`
and injected into the LaMer agents (:mod:`cube_harness.agents.lamer`).

:class:`CrossEpisodeReflector` is the shared mechanism: it owns the file-backed memory and, on a
failed episode, runs the provider's reflection prompt through the agent's (trainable) LLM and
accumulates the lesson. :class:`SimpleReflectionProvider` is the generic baseline (one reward-templated
prompt, plain lesson list, no history parsing); richer, domain-aware providers live in the cubes
(e.g. math-tool-use's ``MathReflectionProvider``, which adds a failure verdict over ``MathAnswer``/
``\\boxed{}``).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from cube.core import ValidatedConfig
from litellm import Message

from cube_harness.llm import LLMCall, LLMConfig, Prompt

logger = logging.getLogger(__name__)

History = list[dict | Message]


class ReflectionProvider(Protocol):
    """Domain-specific surface the generic reflection mechanism delegates to.

    :class:`CrossEpisodeReflector` owns the loop (load/persist memory, call the reflection LLM on
    failure, accumulate lessons); the provider owns everything domain-specific — reading a failed
    attempt out of the agent's history, diagnosing it, and the prompt wording.
    """

    def build_reflection_prompt(
        self, history: History, *, reward: float, max_actions: int, requirements: str | None
    ) -> Prompt:
        """The prompt asking the model to reflect on its just-failed attempt (read from ``history``)."""
        ...

    def format_lessons(self, reflections: list[str], *, max_actions: int) -> str:
        """Render accumulated reflections into a text block to inject into the next attempt."""
        ...

    def extract_lesson(self, raw_reflection: str) -> str:
        """Distil the lesson to store and forward from the raw reflection the LLM produced — e.g.
        pull a tagged conclusion (``<remark>…</remark>``) out of step-by-step reasoning. Identity for
        providers that keep the whole reflection. The full generation is still recorded on the
        trajectory; only what gets carried into the next attempt is distilled."""
        ...


class ReflectionProviderConfig(ValidatedConfig, ABC):
    """Serializable config that produces a :class:`ReflectionProvider` (mirrors ``AgentConfig.make``).

    Lives on the LaMer agent config so it crosses the pickling boundary to Ray workers and is
    Hydra-instantiable (``_target_``); the live provider is built worker-side via :meth:`make`.
    """

    @abstractmethod
    def make(self) -> ReflectionProvider: ...


class CrossEpisodeReflector:
    """Generic, domain-agnostic reflection mechanism shared by the LaMer agents.

    Holds the file-backed memory; on a failed episode (``reward <= 0``) it calls
    ``provider.build_reflection_prompt`` through the agent's (trainable) LLM, appends the reflection,
    and persists. Pure mechanism — all domain content comes from ``provider``.
    """

    def __init__(self, provider: ReflectionProvider, memory_path: Path | None) -> None:
        self._provider = provider
        self._memory_path = memory_path
        self.memory: list[str] = self._load_memory()

    def _load_memory(self) -> list[str]:
        path = self._memory_path
        if path is None or not path.exists():
            return []
        try:
            return json.loads(path.read_text()).get("memory", [])
        except Exception:
            logger.exception("reflector: failed to load memory from %s; starting empty", path)
            return []

    def _persist_memory(self) -> None:
        path = self._memory_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"memory": self.memory}, indent=2))
        except Exception:
            logger.exception("reflector: failed to persist memory to %s", path)

    def lessons(self, *, max_actions: int) -> str | None:
        """The lesson block to inject into the next attempt's prompt, or None if no memory yet."""
        return self._provider.format_lessons(self.memory, max_actions=max_actions) if self.memory else None

    def reflect(
        self,
        *,
        llm: Any,
        llm_config: LLMConfig,
        history: History,
        reward: float,
        max_actions: int,
        requirements: str | None,
    ) -> LLMCall | None:
        """On failure, generate a reflection, append it to memory, persist, and return its ``LLMCall``
        (tag ``"reflection"``) for the trajectory. Success (``reward > 0``) or an empty/failed
        reflection returns ``None`` (memory still persisted)."""
        if reward > 0:
            self._persist_memory()  # success ⇒ no new lesson, but keep the file consistent
            return None
        prompt = self._provider.build_reflection_prompt(
            history, reward=reward, max_actions=max_actions, requirements=requirements
        )
        try:
            resp = llm(prompt)
        except Exception:
            logger.exception("reflection LLM call failed; persisting memory unchanged")
            self._persist_memory()
            return None
        text = (getattr(resp.message, "content", None) or "").strip()
        if text:
            # Store/forward only the distilled lesson (e.g. the <remark> conclusion); the full
            # generation is still recorded below on the LLMCall for the trajectory.
            self.memory.append(self._provider.extract_lesson(text))
        self._persist_memory()
        if not text:
            return None
        logger.info("reflection: %s", (text[:200] + "…") if len(text) > 200 else text)
        return LLMCall(
            tag="reflection",
            llm_config=llm_config,
            prompt=prompt,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", None),
            output_tokens=getattr(resp.usage, "completion_tokens", None),
            output=resp.message,
            usage=resp.usage,
            logprobs=getattr(resp, "logprobs", None),
            completion_token_ids=getattr(resp, "completion_token_ids", None),
            finish_reason=getattr(resp, "finish_reason", None),
            metadata=getattr(resp, "metadata", None),
        )


DEFAULT_REFLECTION_PROMPT = """\
You just attempted this task. The final reward was {reward} (1.0 = success, 0.0 = failure).

In 2-3 sentences of plain prose, describe what worked or didn't, and what concrete
change you would try on the next attempt. Be specific to the actions you took.

No bullets, no headings, no code blocks."""


class SimpleReflectionProvider:
    """Generic, domain-agnostic reflection: one reward-templated prompt, a plain lesson list.

    No history parsing and no failure taxonomy — the baseline the MiniWoB POC uses. Cubes that want
    a richer diagnosis (e.g. a failure verdict) ship their own provider.
    """

    def __init__(self, reflection_prompt: str) -> None:
        self._reflection_prompt = reflection_prompt

    def build_reflection_prompt(
        self, history: History, *, reward: float, max_actions: int, requirements: str | None
    ) -> Prompt:
        return Prompt(
            messages=[
                {"role": "system", "content": "You are an agent reflecting on a finished task attempt."},
                {"role": "user", "content": self._reflection_prompt.format(reward=reward)},
            ]
        )

    def format_lessons(self, reflections: list[str], *, max_actions: int) -> str:
        entries = "\n\n".join(f"### Reflection after attempt {i + 1}\n\n{r}" for i, r in enumerate(reflections))
        return f"## Lessons from previous attempts at this task\n\n{entries}"

    def extract_lesson(self, raw_reflection: str) -> str:
        return raw_reflection  # the whole reflection is the lesson


class SimpleReflectionProviderConfig(ReflectionProviderConfig):
    """Config for :class:`SimpleReflectionProvider`. Override ``reflection_prompt`` to retune."""

    reflection_prompt: str = DEFAULT_REFLECTION_PROMPT

    def make(self) -> ReflectionProvider:
        return SimpleReflectionProvider(self.reflection_prompt)
