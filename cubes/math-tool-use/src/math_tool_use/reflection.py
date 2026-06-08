"""Math-tool-use self-reflection content: the domain ``ReflectionProvider`` for LaMer rollouts.

cube-harness owns the generic reflection *mechanism* (``CrossEpisodeReflector``, the
``ReflectionProvider`` protocol); this module owns the math-specific *content* — the failure verdict
over ``MathAnswer``/``\\boxed{}``, the prompt variants, and how a failed attempt is read out of the
agent's history. Wired into ``LaMerTirAgentConfig`` via :class:`MathReflectionProviderConfig`.

Validated regime (see ~/dev/plans_rl/research/meta_rl_self_reflection): ``max_actions=3`` (tight
budget), ``v1_commit`` chosen on behavioral grounds (reliable finalize + structured reflections).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from cube.core import Action
from litellm import Message

from cube_harness.agents.reflection import History, ReflectionProvider, ReflectionProviderConfig
from cube_harness.core import AgentOutput
from cube_harness.llm import Prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReflectionStrategy:
    """A self-reflection prompt variant. The configurable surface we iterate on.

    ``reflect_system`` / ``reflect_instruction`` shape the reflection the model writes about its own
    failed attempt; ``lesson_header`` / ``commit_reminder`` shape how that reflection is injected into
    the next attempt's prompt. ``reflect_system`` and ``commit_reminder`` may template ``{max_actions}``.

    ``user_template`` (optional) overrides the default reflection user message with a self-contained
    one — placeholders ``{question}`` / ``{reasoning}`` / ``{submitted}`` / ``{max_actions}`` are
    substituted verbatim (so literal ``\\boxed{}`` / ``<remark>`` braces survive). ``conclusion_tag``
    (optional, e.g. ``"remark"``) makes :meth:`MathReflectionProvider.extract_lesson` store only the
    ``<tag>…</tag>`` block, letting the model reason step-by-step but forward only its conclusion.
    """

    name: str
    reflect_system: str
    reflect_instruction: str
    lesson_header: str
    commit_reminder: str = ""
    user_template: str | None = None
    conclusion_tag: str | None = None


REFLECTION_VARIANTS: dict[str, ReflectionStrategy] = {
    # Control: reproduces the original prompt exactly.
    "v0_baseline": ReflectionStrategy(
        name="v0_baseline",
        reflect_system=(
            "You are reflecting on your own failed attempt at a math problem so you can do better "
            "next time. Be concise and concrete. Do NOT solve the problem now."
        ),
        reflect_instruction=(
            "In 2-4 sentences, reflect on what likely went wrong and give yourself concrete, "
            "actionable advice for the next attempt. Do NOT solve the problem now."
        ),
        lesson_header="## Lessons from your previous attempt(s) at THIS problem:",
        commit_reminder="",
    ),
    # v1: kill the dominant failure mode (rabbit-holing under the 3-action budget) and the
    # secondary one (over-correcting a sound approach onto a wrong answer). Chosen LaMer scaffold.
    "v1_commit": ReflectionStrategy(
        name="v1_commit",
        reflect_system=(
            "You are reviewing your own failed attempt at a math problem to do better on the very "
            "next try, under a STRICT budget of only {max_actions} actions per attempt (typically: "
            "one run_python_code, then MathAnswer). Be concise and concrete. Diagnose only — do NOT "
            "solve the problem now."
        ),
        reflect_instruction=(
            "In 2-3 sentences, give your next attempt concrete, actionable advice. First, name the "
            "single most likely mistake in the previous attempt. Then decide: if the overall method "
            "was sound, say which ONE step to fix; if the method was flawed, name a specific "
            "different method to try — but do NOT abandon a sound approach just because the answer "
            "was unverified. Do NOT solve the problem now."
        ),
        lesson_header="## Review of your previous attempt(s) at THIS problem — apply it, then COMMIT an answer:",
        commit_reminder=(
            "You have only {max_actions} actions this attempt. Do NOT re-derive everything from "
            "scratch: verify the one critical step with a single run_python_code if needed, then you "
            "MUST finalize with MathAnswer using \\boxed{<answer>}. Submitting your best answer "
            "always beats running out of actions with no answer submitted."
        ),
    ),
    # v2: keep v1's commit discipline, but fix the residual answer-ANCHORING (v1 was too sticky and
    # resubmitted a known-wrong answer across attempts). Force a concrete change + a new answer.
    # NOTE: on held-out validation v2 regressed vs v1 (re-introduced over-correction); kept for
    # the record / ablations, not the default.
    "v2_explore": ReflectionStrategy(
        name="v2_explore",
        reflect_system=(
            "You are reviewing your own failed attempt at a math problem to do better on the very "
            "next try, under a STRICT budget of only {max_actions} actions per attempt (typically: "
            "one run_python_code, then MathAnswer). Be concise and concrete. Diagnose only — do NOT "
            "solve the problem now."
        ),
        reflect_instruction=(
            "In 2-3 sentences, give your next attempt concrete, actionable advice. (1) Name the single "
            "most likely error. (2) If a specific answer was already submitted and marked INCORRECT, "
            "state that exact value and tell yourself NOT to submit it again — a wrong answer means a "
            "step or the whole method is wrong, so name a *materially different* method or the specific "
            "computation to redo differently; do not just restate the same approach and the same number. "
            "(3) If instead no valid answer was ever produced, focus on reaching a submittable answer. "
            "Do NOT solve the problem now."
        ),
        lesson_header="## Review of your previous attempt(s) at THIS problem — change something concrete, then COMMIT a NEW answer:",
        commit_reminder=(
            "You have only {max_actions} actions this attempt. Do NOT re-derive everything from "
            "scratch and do NOT resubmit an answer already shown to be wrong: change one concrete "
            "thing, verify the key step with a single run_python_code if needed, then you MUST finalize "
            "with MathAnswer using \\boxed{<answer>}. Submitting your best NEW answer always beats "
            "running out of actions or repeating a known-wrong answer."
        ),
    ),
    # v3: adheres closely to the LaMer paper's reflection prompt — the model reasons step-by-step
    # about its past attempt, then commits an improved plan inside <remark>…</remark>; only the remark
    # is forwarded as the lesson (conclusion_tag). Self-diagnose by construction (ignores use_verdict).
    "v3_lamer": ReflectionStrategy(
        name="v3_lamer",
        reflect_system=(
            "You are an expert agent solving competition mathematics problems with a Python tool, under "
            "a STRICT budget of only {max_actions} actions per attempt (typically one run_python_code to "
            "compute or verify, then MathAnswer to finalize the answer in LaTeX \\boxed{} form).\n\n"
            "You will be given the history of a past attempt. Your job is to reflect on that attempt, "
            "identify any mistakes or inefficiencies, and devise a concise, improved plan for the next "
            "attempt at the SAME problem."
        ),
        # Unused for this variant (user_template carries the instruction inline); kept for documentation.
        reflect_instruction=(
            "Reason step-by-step about what went wrong, then give a concise improved plan inside "
            "<remark></remark> tags."
        ),
        lesson_header="## Reflection and improved plan from your previous attempt(s) at THIS problem:",
        commit_reminder=(
            "You have only {max_actions} actions this attempt. Follow the plan above, verify the one "
            "critical step with a single run_python_code if needed, then you MUST finalize with "
            "MathAnswer using \\boxed{<answer>}."
        ),
        user_template=(
            "# Past Experience\n"
            "The problem is:\n{question}\n\n"
            "The approach you took (your reasoning; may be truncated):\n{reasoning}\n\n"
            "Your submitted final answer: {submitted}\n\n"
            "The task is NOT successfully completed.\n\n"
            "Now it's your turn to reflect on the past experience and come up with a new plan of action.\n"
            "- Your response should first be step-by-step reasoning about the strategy and path you took "
            "to attempt the problem. Identify where things went wrong or could be better.\n"
            "- Then devise a concise, new plan of action that accounts for your mistake, with reference to "
            "the specific steps you should have taken (and remember to finalize by calling MathAnswer with "
            "the answer in \\boxed{} form).\n"
            "- Finally, end the response with your reflection and improved plan inside <remark></remark> "
            "tags, to guide the next trial."
        ),
        conclusion_tag="remark",
    ),
}

# Used when ``use_verdict=False``: the reflector is NOT handed the failure category, so the
# instruction must not presume the failure is method-based (v1_commit's does). It points the model
# at the requirements block + transcript and lets it discover the failure (incl. format) itself.
_SELF_DIAGNOSE_INSTRUCTION = (
    "In 2-3 sentences, diagnose the single most likely reason the attempt above failed and give your "
    "next attempt concrete, actionable advice. Check both your reasoning AND whether you met the "
    "requirements stated above (e.g. how to finalize and format the answer). Do NOT solve the problem now."
)


def _msg_attr(msg: dict | Message, key: str) -> Any:
    """Read ``key`` from a history entry that may be a dict or a litellm ``Message``."""
    return msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)


def summarize_history(history: History) -> tuple[str, str | None, str, int]:
    """(question, submitted_MathAnswer, reasoning_excerpt, num_python_calls) from agent history."""
    question = ""
    submitted: str | None = None
    reasoning_parts: list[str] = []
    n_py = 0
    for msg in history:
        role = _msg_attr(msg, "role")
        content = _msg_attr(msg, "content")
        if role == "user" and not question and isinstance(content, str):
            question = content
        if role == "assistant" and isinstance(content, str) and content.strip():
            reasoning_parts.append(content.strip())
        for tc in _msg_attr(msg, "tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) if fn else None
            if name == "run_python_code":
                n_py += 1
            elif name == "MathAnswer":
                raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except Exception:
                    args = {}
                ans = args.get("answer") if isinstance(args, dict) else None
                if isinstance(ans, str) and ans.strip():
                    submitted = ans.strip()
    reasoning = "\n".join(reasoning_parts)
    if len(reasoning) > 1200:
        reasoning = "...(truncated)\n" + reasoning[-1200:]
    return question or "(the problem stated above)", submitted, reasoning, n_py


def extract_attempt_summary(trajectory: Any) -> tuple[str | None, str]:
    """Return (submitted_answer, reasoning_excerpt) from an episode *trajectory* (for the probe scorer).

    submitted_answer: the last ``MathAnswer`` action argument, or None if never submitted.
    reasoning_excerpt: concatenated assistant message text (tail-truncated).
    """
    submitted: str | None = None
    reasoning_parts: list[str] = []
    for step in trajectory.steps:
        if not isinstance(step.output, AgentOutput):
            continue
        for action in step.output.actions:
            if isinstance(action, Action) and action.name == "MathAnswer":
                ans = action.arguments.get("answer")
                if isinstance(ans, str) and ans.strip():
                    submitted = ans.strip()
        for call in step.output.llm_calls:
            content = getattr(call.output, "content", None)
            if isinstance(content, str) and content.strip():
                reasoning_parts.append(content.strip())
    reasoning = "\n".join(reasoning_parts)
    if len(reasoning) > 1200:
        reasoning = "...(truncated)\n" + reasoning[-1200:]
    return submitted, reasoning


def build_verdict(reward_info: dict, submitted: str | None, max_actions: int) -> str:
    """A specific, actionable description of how the previous attempt failed.

    Distinguishes the failure modes the reward actually penalises: never calling MathAnswer, calling
    it without the required \\boxed{} LaTeX, an unparsable answer, or a parseable-but-wrong answer.
    The first two are read straight from ``submitted`` (gold-independent), so the verdict is correct
    even on the training path where ``reward_info`` carries only ``num_python_calls``; only the
    unparsable branch needs the cube's ``no_error`` flag and degrades to "wrong" without it.
    """
    n_py = reward_info.get("num_python_calls", 0)
    if submitted is None:
        return (
            f"You never called MathAnswer (you made {n_py} python call(s) and exhausted your "
            f"{max_actions}-action budget). You MUST finalize by calling MathAnswer."
        )
    if submitted.rfind("\\boxed{") < 0:
        return (
            f"You called MathAnswer with {submitted!r}, but it was NOT wrapped in the required "
            "\\boxed{...} LaTeX, so it did not count. Always submit \\boxed{<answer>}."
        )
    if not reward_info.get("no_error", True):
        return (
            f"You called MathAnswer with {submitted!r}, but it could not be parsed as a valid "
            "answer. Submit a clean \\boxed{<answer>}."
        )
    return (
        f"You submitted {submitted!r} via MathAnswer, but the answer was INCORRECT. "
        "Re-derive carefully and verify with run_python_code before finalizing."
    )


class MathReflectionProvider:
    """Math ``ReflectionProvider``: a failure verdict + a chosen prompt variant over the math TIR
    protocol (``MathAnswer``/``\\boxed{}``/``run_python_code``)."""

    def __init__(self, strategy: ReflectionStrategy, use_verdict: bool) -> None:
        self._strategy = strategy
        self._use_verdict = use_verdict

    def build_reflection_prompt(
        self, history: History, *, reward: float, max_actions: int, requirements: str | None
    ) -> Prompt:
        question, submitted, reasoning, n_py = summarize_history(history)
        system = self._strategy.reflect_system.replace("{max_actions}", str(max_actions))
        if self._strategy.user_template is not None:
            # Self-contained template (e.g. v3_lamer). .replace (not .format) so literal \boxed{} /
            # <remark> braces survive. Self-diagnose by construction — use_verdict does not apply.
            user = (
                self._strategy.user_template.replace("{question}", question)
                .replace("{reasoning}", reasoning or "(none captured)")
                .replace("{submitted}", submitted or "(none — you did not call MathAnswer)")
                .replace("{max_actions}", str(max_actions))
            )
            return Prompt(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self._use_verdict:
            # reward_info isn't available here (reflection is an agent concern); pass num_python_calls
            # so the "never submitted" branch is precise — the boxed/unboxed split is read from submitted.
            verdict = build_verdict({"num_python_calls": n_py}, submitted, max_actions)
            failure = f"Your previous attempt failed. {verdict}"
            req_block = ""
            instruction = self._strategy.reflect_instruction
        else:
            failure = "Your previous attempt failed (it received no reward)."
            req_block = f"The requirements you must satisfy:\n{requirements}\n\n" if requirements else ""
            instruction = _SELF_DIAGNOSE_INSTRUCTION
        user = (
            f"You are attempting this math problem:\n\n{question}\n\n"
            f"{req_block}{failure}\n\n"
            f"Reasoning from your previous attempt (may be truncated):\n{reasoning or '(none captured)'}\n\n"
            f"{instruction}"
        )
        return Prompt(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])

    def format_lessons(self, reflections: list[str], *, max_actions: int) -> str:
        lessons = "\n".join(f"- Attempt {i + 1} reflection: {r}" for i, r in enumerate(reflections))
        parts = [f"{self._strategy.lesson_header}\n{lessons}"]
        if self._strategy.commit_reminder:
            parts.append(self._strategy.commit_reminder.replace("{max_actions}", str(max_actions)))
        return "\n\n".join(parts)

    def extract_lesson(self, raw_reflection: str) -> str:
        """For a variant with a ``conclusion_tag`` (e.g. ``remark``), store only the last
        ``<tag>…</tag>`` block — the model reasons step-by-step but forwards only its conclusion.
        Falls back to the full reflection if the tag is absent. Identity otherwise."""
        tag = self._strategy.conclusion_tag
        if not tag:
            return raw_reflection
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", raw_reflection, re.DOTALL)
        if not matches:
            logger.warning("reflection: no <%s> block in output; forwarding the full reflection", tag)
            return raw_reflection
        return matches[-1].strip()


class MathReflectionProviderConfig(ReflectionProviderConfig):
    """Config for :class:`MathReflectionProvider`. ``reflection_variant`` selects a prompt strategy
    from :data:`REFLECTION_VARIANTS`. ``use_verdict`` defaults to False: the reflector is NOT handed a
    hand-built failure diagnosis and must diagnose failures itself (given the task + format contract) —
    the meta-RL stance that it should learn to surface the right hint, including format ones. Set
    ``use_verdict=True`` to inject the hand-built verdict instead."""

    reflection_variant: str = "v1_commit"
    use_verdict: bool = False

    def make(self) -> ReflectionProvider:
        if self.reflection_variant not in REFLECTION_VARIANTS:
            raise ValueError(
                f"unknown reflection_variant {self.reflection_variant!r}; choose from {list(REFLECTION_VARIANTS)}"
            )
        return MathReflectionProvider(REFLECTION_VARIANTS[self.reflection_variant], self.use_verdict)
