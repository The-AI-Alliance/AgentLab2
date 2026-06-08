"""Tests for the reflection mechanism + providers (``cube_harness.agents.reflection``).

The generic pieces (``CrossEpisodeReflector``, ``SimpleReflectionProvider``) are tested directly; the
math ``MathReflectionProvider`` is tested behind an importorskip (workspace cubes aren't installed in
the default CI), covering the verdict's gold-independent boxed check and the use_verdict toggle.
"""

from pathlib import Path
from typing import Any

import pytest
from litellm import Message

from cube_harness.agents.reflection import (
    CrossEpisodeReflector,
    SimpleReflectionProvider,
    SimpleReflectionProviderConfig,
)
from cube_harness.llm import LLMConfig, LLMResponse, Prompt, Usage

_LLM_CONFIG = LLMConfig(model_name="openai/gpt-4o")


def _fake_llm(content: str):
    def _call(_prompt: Any) -> LLMResponse:
        return LLMResponse(
            message=Message(role="assistant", content=content),
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cached_tokens=0,
                cache_creation_tokens=0,
                cost=0.0,
            ),
        )

    return _call


def _raising_llm(_prompt: Any) -> LLMResponse:
    raise RuntimeError("provider down")


class _RemarkProvider:
    """Fake provider whose extract_lesson keeps only <remark>…</remark> — to test reflector wiring."""

    def build_reflection_prompt(self, history: Any, *, reward: float, max_actions: int, requirements: Any) -> Prompt:
        return Prompt(messages=[{"role": "user", "content": "reflect"}])

    def format_lessons(self, reflections: list[str], *, max_actions: int) -> str:
        return "; ".join(reflections)

    def extract_lesson(self, raw: str) -> str:
        if "<remark>" in raw and "</remark>" in raw:
            return raw.split("<remark>", 1)[1].split("</remark>", 1)[0].strip()
        return raw


# ---------------------------------------------------------------------------
# SimpleReflectionProvider
# ---------------------------------------------------------------------------


class TestSimpleReflectionProvider:
    def test_build_prompt_interpolates_reward(self) -> None:
        provider = SimpleReflectionProviderConfig().make()
        prompt = provider.build_reflection_prompt([], reward=0.0, max_actions=5, requirements=None)
        user = [m for m in prompt.messages if m["role"] == "user"][0]["content"]
        assert "0.0" in user

    def test_format_lessons_numbers_attempts(self) -> None:
        provider = SimpleReflectionProvider("ignored")
        block = provider.format_lessons(["a", "b"], max_actions=3)
        assert "## Lessons from previous attempts at this task" in block
        assert "Reflection after attempt 1" in block and "Reflection after attempt 2" in block


# ---------------------------------------------------------------------------
# CrossEpisodeReflector — the shared mechanism (reward gate, persistence, failure handling)
# ---------------------------------------------------------------------------


class TestCrossEpisodeReflector:
    def _reflector(self, path: Path | None) -> CrossEpisodeReflector:
        return CrossEpisodeReflector(SimpleReflectionProviderConfig().make(), path)

    def test_success_skips_reflection(self, tmp_path: Path) -> None:
        r = self._reflector(tmp_path / "m.json")
        call = r.reflect(
            llm=_fake_llm("nope"),
            llm_config=_LLM_CONFIG,
            history=[],
            reward=1.0,
            max_actions=3,
            requirements=None,
        )
        assert call is None
        assert r.memory == []

    def test_failure_reflects_and_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        r = self._reflector(path)
        call = r.reflect(
            llm=_fake_llm("a lesson"),
            llm_config=_LLM_CONFIG,
            history=[],
            reward=0.0,
            max_actions=3,
            requirements=None,
        )
        assert call is not None and call.tag == "reflection"
        assert r.memory == ["a lesson"]
        # Fresh reflector reads it back.
        assert CrossEpisodeReflector(SimpleReflectionProviderConfig().make(), path).memory == ["a lesson"]

    def test_empty_reflection_not_appended(self, tmp_path: Path) -> None:
        r = self._reflector(tmp_path / "m.json")
        assert (
            r.reflect(
                llm=_fake_llm("  "), llm_config=_LLM_CONFIG, history=[], reward=0.0, max_actions=3, requirements=None
            )
            is None
        )
        assert r.memory == []

    def test_llm_failure_swallowed(self, tmp_path: Path) -> None:
        r = self._reflector(tmp_path / "m.json")
        assert (
            r.reflect(
                llm=_raising_llm, llm_config=_LLM_CONFIG, history=[], reward=0.0, max_actions=3, requirements=None
            )
            is None
        )
        assert r.memory == []

    def test_lessons_none_when_empty(self, tmp_path: Path) -> None:
        assert self._reflector(tmp_path / "m.json").lessons(max_actions=3) is None

    def test_reflect_stores_extracted_lesson_keeps_full_on_call(self, tmp_path: Path) -> None:
        r = CrossEpisodeReflector(_RemarkProvider(), tmp_path / "m.json")
        call = r.reflect(
            llm=_fake_llm("step by step reasoning... <remark>USE THIS PLAN</remark>"),
            llm_config=_LLM_CONFIG,
            history=[],
            reward=0.0,
            max_actions=3,
            requirements=None,
        )
        assert r.memory == ["USE THIS PLAN"]  # only the distilled conclusion is forwarded
        assert call is not None and "step by step reasoning" in (call.output.content or "")  # full kept


# ---------------------------------------------------------------------------
# MathReflectionProvider (math-tool-use) — verdict correctness + use_verdict toggle
# ---------------------------------------------------------------------------

try:
    import math_tool_use as math
except ImportError:  # workspace cubes aren't installed in the default CI
    math = None

_needs_math = pytest.mark.skipif(math is None, reason="math_tool_use not installed")


def _math_history(submitted: str) -> list[dict]:
    return [
        {"role": "user", "content": "What is the answer?"},
        {
            "role": "assistant",
            "content": "I think it is right.",
            "tool_calls": [{"function": {"name": "MathAnswer", "arguments": f'{{"answer": "{submitted}"}}'}}],
        },
    ]


@_needs_math
class TestBuildVerdict:
    def test_unboxed_answer_flags_format(self) -> None:
        # The gold-independent boxed check: correct value, but submitted without \boxed{}.
        v = math.build_verdict({"num_python_calls": 0}, "50648", 3)
        assert "NOT wrapped" in v

    def test_boxed_but_wrong_is_incorrect(self) -> None:
        v = math.build_verdict({}, "\\boxed{999}", 3)
        assert "INCORRECT" in v

    def test_never_submitted(self) -> None:
        v = math.build_verdict({"num_python_calls": 2}, None, 3)
        assert "never called MathAnswer" in v


@_needs_math
class TestMathReflectionProvider:
    def test_default_is_self_diagnose(self) -> None:
        assert math.MathReflectionProviderConfig().use_verdict is False

    def test_verdict_on_diagnoses_format(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v1_commit", use_verdict=True).make()
        prompt = provider.build_reflection_prompt(_math_history("50648"), reward=0.0, max_actions=3, requirements="REQ")
        user = prompt.messages[1]["content"]
        assert "NOT wrapped" in user

    def test_verdict_off_drops_verdict_and_adds_requirements(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v1_commit", use_verdict=False).make()
        prompt = provider.build_reflection_prompt(_math_history("50648"), reward=0.0, max_actions=3, requirements="REQ")
        user = prompt.messages[1]["content"]
        assert "NOT wrapped" not in user  # no hand-built verdict
        assert "REQ" in user  # format contract is surfaced instead
        assert "diagnose the single most likely reason" in user  # neutral self-diagnose instruction

    def test_format_lessons_includes_commit_reminder(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v1_commit").make()
        block = provider.format_lessons(["verify first"], max_actions=3)
        assert "verify first" in block
        assert "MUST finalize with MathAnswer" in block  # v1_commit commit_reminder

    def test_lamer_variant_uses_remark_template(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v3_lamer").make()
        user = provider.build_reflection_prompt(
            _math_history("50648"), reward=0.0, max_actions=3, requirements="REQ"
        ).messages[1]["content"]
        assert "# Past Experience" in user
        assert "step-by-step" in user
        assert "<remark></remark>" in user
        assert "You are attempting this math problem:" not in user  # not the default template

    def test_lamer_extract_keeps_only_remark(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v3_lamer").make()
        raw = "Let me think step by step. I made an error.\n<remark>Recompute step 2 with sympy, then box it.</remark>"
        assert provider.extract_lesson(raw) == "Recompute step 2 with sympy, then box it."

    def test_lamer_extract_falls_back_without_remark(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v3_lamer").make()
        assert provider.extract_lesson("no tags here") == "no tags here"

    def test_nonlamer_extract_is_identity(self) -> None:
        provider = math.MathReflectionProviderConfig(reflection_variant="v1_commit").make()
        assert provider.extract_lesson("anything <remark>x</remark>") == "anything <remark>x</remark>"
