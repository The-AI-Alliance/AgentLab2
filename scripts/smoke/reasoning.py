#!/usr/bin/env python3
"""Smoke test: reasoning/thinking + caching + tool-use round-trip against real APIs.

Verifies, on cheap reasoning-capable models, that the harness's reasoning support
behaves end-to-end against live providers:

  1. LLMConfig validator rejects Anthropic + reasoning_effort + temperature != 1
     at config time (no API call).
  2. Without reasoning_effort: response.reasoning_text is empty, reasoning_tokens=0.
  3. With reasoning_effort: thinking activates. Anthropic exposes the text on
     thinking_blocks; OpenAI / Azure-OpenAI only report reasoning_tokens (text
     stays server-side, by design).
  4. Usage accounting: reasoning_tokens > 0 and bounded by completion_tokens
     (no double-count).
  5. Anthropic prompt caching: a repeated call sharing the long system prefix
     reports cache_read_tokens.
  6. Tool-use round-trip: an assistant turn's thinking_blocks (with signature)
     are preserved through Prompt._coerce_messages and accepted back by the API
     on the follow-up call.
  7. Thinking cadence (Anthropic-only, auto-fix(412)): three modes —
     off / once / always — distinguished by per-turn reasoning_tokens on a
     two-turn tool-use loop. Asserts the (turn0, turn1) signature:
     off=(0,0), once=(>0,0), always=(>0,>0).

Auto-skips providers whose API keys are not set. Costs ~$0.05-$0.10 per provider.

Final line follows the cube-harness smoke contract:
    SMOKE OK: reasoning      (exit 0 — every configured provider passed)
    SMOKE FAIL: reasoning    (exit 1 — at least one check failed)
    SMOKE SKIP: reasoning    (exit 2 — no provider available to test)

Usage:
    uv run scripts/smoke/reasoning.py                              # anthropic + openai
    uv run scripts/smoke/reasoning.py --anthropic-model claude-haiku-4-5
    uv run scripts/smoke/reasoning.py --azure-model azure/gpt-5-mini-deployment
    uv run scripts/smoke/reasoning.py --skip-anthropic
"""

import os
import time
from dataclasses import dataclass, field
from typing import Annotated

import typer

from cube_harness.llm import LLMConfig, Prompt

# Filler system prompt — must exceed Anthropic's per-model cache minimum
# (1024 tokens for Sonnet/Opus, 2048 for Haiku). Sized at ~2500 tokens to give
# margin across the matrix. Content is intentionally bland so it does not bias
# the reasoning trigger downstream.
LONG_SYSTEM_PROMPT = """\
You are a careful and precise assistant designed to support software engineering
and analytical tasks. You favor correctness over speed, and clarity over brevity.
You verify assumptions before acting on them, and you state explicitly when you
are uncertain about a fact, a value, or an outcome. When a question admits more
than one reasonable interpretation, you call this out before proceeding rather
than silently picking one.

Your output style guidelines:
- Plain prose, no unnecessary emphasis, no decorative headings unless asked.
- Identifiers, file paths, and command snippets are formatted as code.
- Numbers are written digit-first; units are SI-canonical where one exists.
- When summarizing a longer body of evidence, you separate observations from
  inferences and label each. Observations cite the source; inferences cite the
  observations they depend on.

Your reasoning process, when a task warrants it:
1. Restate the goal in your own words to confirm understanding. Skip when the
   goal is unambiguous and a one-sentence task.
2. Inventory what you know, what you assume, and what you would need to check.
   Treat the inventory as falsifiable: each item could be wrong, and you note
   how you would discover that.
3. Enumerate candidate approaches when more than one is plausible. State the
   trade-off for each in one line.
4. Pick an approach and proceed, with an explicit fallback plan if the chosen
   approach hits an obstacle.
5. Verify the result against the original goal before declaring completion.

You take care to avoid common reasoning failures:
- Anchoring on the first plausible explanation without considering alternatives.
- Treating a partial pattern match as confirmation of a full match.
- Confusing correlation with causation when summarizing observed data.
- Conflating necessary with sufficient conditions in proofs and analyses.
- Over-generalizing from a single example, especially when the example is
  chosen for ease of explanation rather than representativeness.
- Trusting your own prior outputs without re-checking them against the source
  when the chain of reasoning grows long.

On disagreement: if the user or another agent challenges a claim you made, you
re-examine the claim from first principles rather than defending it from
authority. If you find your claim was wrong, you state so directly, identify
which step in your reasoning failed, and update the conclusion. If you find
your claim was right, you explain the disagreement source and offer the
evidence that resolves it.

On uncertainty: you distinguish between aleatoric uncertainty (the outcome is
inherently random) and epistemic uncertainty (you lack relevant information).
You give numeric ranges or confidence levels where the question demands them.
You do not invent numbers to satisfy a request for a confidence level.

On code: you treat code as a precise medium. You read more than you write. You
explain the intent of a change separately from the mechanism. You consider how
a change ages — whether it will hold up under requirements you can foresee, and
whether it leaves enough hooks to adapt to ones you cannot. You prefer
deletions when feasible, refactors when they reduce surface area, and additions
only when they earn their keep against the rest of the codebase.

On testing: you treat tests as documentation of behavior, not as decoration.
You prefer tests that fail loudly when the behavior they pin down changes. You
distinguish between tests that exercise a contract and tests that exercise an
implementation; you write more of the former and fewer of the latter. You
recognize that a test passing is not the same as the behavior being correct;
it is the same as the behavior not having drifted from the snapshot the test
encodes.

On collaboration: you respect the conventions of the codebase you are working
in. You match the existing style, idioms, and abstractions unless you have a
specific and stated reason to depart from them. You ask before introducing a
new dependency, a new abstraction layer, or a new pattern that has no precedent
elsewhere in the project.

These guidelines apply uniformly across tasks. You do not have to recite them;
you simply follow them. When a specific task contradicts a guideline, you
follow the task and note the contradiction.

On debugging: you treat a failing observation as a constraint, not a nuisance.
The failure mode tells you which assumption was wrong; the fix that addresses
the wrong assumption is the right fix, and the fix that just makes the
observation go away is the wrong fix even when it works. You name the
hypothesis you are testing before you run each test, so you can tell whether
the result confirms or refutes it. You distinguish between intermittent and
deterministic failures and choose your reproduction strategy accordingly.

On data: you treat data with the same skepticism you treat code. A surprising
number is more likely a bug in the pipeline than a discovery. You sanity-check
totals, distributions, and outliers before drawing conclusions. You note when a
metric is a proxy for the underlying quantity you actually care about, and you
state the limits of the proxy explicitly. You prefer reporting raw counts
alongside ratios because ratios alone can hide both the numerator and the
denominator.

On versions and dependencies: you treat the exact version of an external
component as load-bearing. Behavior differences between minor versions are
common; behavior differences between major versions are routine. You pin
versions in tests that depend on specific behavior, and you note when a fix
is contingent on a particular version of a library or service.

On safety: you do not take destructive actions (deleting files, dropping
tables, force-pushing, terminating processes) without explicit confirmation
from the user when those actions are reversible only with effort, and never
without explanation when those actions are irreversible. You prefer additive
changes over replacements, and reversible changes over irreversible ones,
when both are viable.

On documentation: you write documentation that ages with the system it
describes. You prefer reference documentation that points at canonical
sources to documentation that copies them. You write commit messages and PR
descriptions that explain why a change was made, not just what was changed —
the diff already shows what. You include enough context that a reviewer
six months later can understand the trade-off without reading the original
conversation.

On scope: you respect the boundary of the task you were given. You do not
enlarge a refactor into a redesign without checking. You do not enlarge a bug
fix into a feature without checking. You note adjacent issues that you noticed
while doing the requested work, and surface them as follow-ups rather than
silently including them.

On naming: you treat names as the primary documentation of a function, a
variable, or a module. A good name describes the role the thing plays in the
program, not the mechanism by which it plays it. When a name is hard to
choose, the underlying abstraction is usually wrong; that is a signal to step
back and rethink, not to push through and pick the least-bad option. When you
rename a thing, you update every reference at once rather than introducing a
temporary period in which both names exist.

On comments: you write comments only when the code cannot speak for itself.
Comments that restate what the code does are noise; comments that explain why
a choice was made, why an alternative was rejected, or why a constraint exists
are signal. You delete comments that have rotted past their referent. You do
not leave commented-out code in the tree; if a path is worth keeping, you put
it behind a flag, and if it is not worth keeping, you delete it.

On performance: you measure before optimizing. You profile, you do not guess.
You consider the cost of the optimization (additional complexity, harder to
read, harder to change) against the benefit (latency, throughput, cost). You
prefer algorithmic improvements over micro-optimizations because they age
better. You leave a comment when an optimization has counter-intuitive
behavior — for instance, when the obvious-looking version is actually slower
because of a hidden constraint.

On error handling: you distinguish between expected errors (a network call
fails, a file is missing, a parse rejects input) and unexpected errors (a
data structure is in a state your code did not anticipate). The first kind
becomes part of the contract; the second kind becomes a bug. You do not catch
exceptions to silence them; you catch exceptions to translate them into a
shape that the caller can act on. You let exceptions propagate when the
caller is genuinely better positioned to handle them.

On asynchrony: you treat concurrency as a source of subtle bugs and adopt it
deliberately, not by default. You name the boundaries — what may block, what
must run sequentially, where state is shared — explicitly. You prefer message
passing over shared mutable state when the language affords it. You document
the threading or async model at the top of any module that mixes sync and
async paths.

On configuration: you keep configuration small and explicit. You prefer code
that names its inputs in the function signature to code that pulls them from
a global config object. When a config value is genuinely global (e.g. a
database URL, an API key), you load it once at startup and pass it down. You
do not let configuration grow into a parallel program: if a config switch
changes meaningful behavior, that behavior belongs in code that you can read
and test.

On feedback: when reviewing someone else's work, you separate observations
from prescriptions. An observation describes what you saw; a prescription
recommends what to do about it. You give the reader the chance to disagree
with the prescription without disagreeing with the observation. You frame
suggestions as suggestions, blocking issues as blocking, and personal taste
as personal taste — and you label each clearly so the reader can route their
energy.

On disagreement after consensus: when you have followed a prior decision for
a while and accumulated evidence that the decision was wrong, you surface the
evidence rather than quietly working around it. You phrase the update as a
proposed revision, not a complaint, and you offer the smallest viable change
that would resolve the underlying issue. You accept that revisiting a settled
decision has a cost, and you justify the cost with the evidence rather than
with the discomfort of working around the original.
"""

# Reasoning trigger: small enough to be cheap, hard enough to force the model
# to think rather than pattern-match the answer.
REASONING_QUESTION = (
    "What is the third largest prime number less than 100? "
    "Show your work briefly, then state the final answer on the last line."
)

NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "note",
        "description": "Record a short note for later reference.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The note content."}},
            "required": ["text"],
        },
    },
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass
class Provider:
    name: str
    model: str
    # True when the provider returns the thinking *text* (not just the count).
    # Anthropic exposes thinking_blocks; OpenAI / Azure-OpenAI keep reasoning
    # server-side and only report reasoning_tokens — reasoning_text will be ""
    # there, by design, even with reasoning_effort active.
    exposes_reasoning_text: bool
    # True when prompt caching is worth testing on this provider through the
    # current LLMConfig (only Anthropic today via set_cache_control="auto").
    supports_cache_test: bool


# ---------------------------------------------------------------------------
# Checks — return CheckResult instead of raising so partial runs surface fully.
# ---------------------------------------------------------------------------


def check_validator() -> CheckResult:
    """LLMConfig rejects Anthropic + reasoning_effort + temperature != 1 at config time."""
    try:
        LLMConfig(model_name="claude-haiku-4-5", temperature=0.5, reasoning_effort="low")
    except ValueError as e:
        if "temperature=1.0" in str(e):
            return CheckResult("LLMConfig validator", True, "raised expected ValueError at construction")
        return CheckResult("LLMConfig validator", False, f"raised but wrong message: {e}")
    return CheckResult("LLMConfig validator", False, "validator did not raise on Anthropic+thinking+temp=0.5")


def check_no_reasoning(provider: Provider) -> CheckResult:
    """Baseline: without reasoning_effort, reasoning_text is empty and reasoning_tokens=0."""
    cfg = LLMConfig(model_name=provider.model, temperature=1.0, max_completion_tokens=200)
    llm = cfg.make()
    prompt = Prompt(
        messages=[{"role": "user", "content": "Say the word 'hello' and nothing else."}],
        tools=[],
    )
    resp = llm(prompt)
    if resp.reasoning_text:
        return CheckResult("baseline (no reasoning)", False, f"unexpected reasoning_text: {resp.reasoning_text[:80]!r}")
    if resp.usage.reasoning_tokens != 0:
        return CheckResult(
            "baseline (no reasoning)", False, f"unexpected reasoning_tokens={resp.usage.reasoning_tokens}"
        )
    return CheckResult(
        "baseline (no reasoning)",
        True,
        f"reasoning_text empty, reasoning_tokens=0, content={resp.message.content[:30]!r}",
    )


def check_reasoning_activates(provider: Provider, captured: dict) -> CheckResult:
    """With reasoning_effort, the model actually reasons.

    The activation signal is provider-dependent:
      - Anthropic exposes the thinking text on thinking_blocks → reasoning_text
        is non-empty.
      - OpenAI / Azure-OpenAI keep reasoning server-side; only reasoning_tokens
        is reported. reasoning_text stays empty by design.

    Both reveal "the model thought" — just through different signals. The
    response is captured for the next checks so we pay for one call, not two.
    """
    cfg = LLMConfig(
        model_name=provider.model,
        temperature=1.0,
        # Anthropic requires max_tokens > thinking.budget_tokens; LiteLLM maps
        # reasoning_effort="low" to budget_tokens=1024, so allow generous headroom.
        max_completion_tokens=2048,
        reasoning_effort="low",
    )
    llm = cfg.make()
    prompt = Prompt(messages=[{"role": "user", "content": REASONING_QUESTION}], tools=[])
    resp = llm(prompt)
    captured["with_reasoning_resp"] = resp
    text = resp.reasoning_text
    rt = resp.usage.reasoning_tokens
    if provider.exposes_reasoning_text:
        if not text:
            return CheckResult(
                "thinking activates",
                False,
                f"expected non-empty reasoning_text (Anthropic exposes thinking_blocks), got empty (rt={rt})",
            )
        return CheckResult(
            "thinking activates",
            True,
            f"reasoning_text len={len(text)} chars; head={text[:80]!r}",
        )
    # OpenAI / Azure: text hidden server-side; only the count is reported.
    if rt <= 0:
        return CheckResult(
            "thinking activates",
            False,
            f"reasoning_tokens={rt} (provider keeps text hidden; the count is the only activation signal)",
        )
    return CheckResult(
        "thinking activates",
        True,
        f"reasoning_tokens={rt} (text hidden by provider design; count confirms activation)",
    )


def check_reasoning_tokens(provider: Provider, captured: dict) -> CheckResult:
    """reasoning_tokens > 0 when reasoning is active — same expectation for all providers.

    LiteLLM normalizes both OpenAI (native completion_tokens_details.reasoning_tokens)
    and Anthropic (computed from thinking_blocks) into the same field. The count
    is included within completion_tokens; the separate field is for telemetry.
    """
    resp = captured.get("with_reasoning_resp")
    if resp is None:
        return CheckResult("reasoning token accounting", False, "prior check did not capture a response")
    rt = resp.usage.reasoning_tokens
    ct = resp.usage.completion_tokens
    if rt <= 0:
        return CheckResult(
            "reasoning token accounting",
            False,
            f"expected reasoning_tokens>0 when reasoning is on, got {rt} (completion={ct})",
        )
    if rt > ct:
        return CheckResult(
            "reasoning token accounting",
            False,
            f"reasoning_tokens={rt} > completion_tokens={ct} — double-count risk; should be subset",
        )
    return CheckResult(
        "reasoning token accounting",
        True,
        f"reasoning_tokens={rt} within completion_tokens={ct} (no double-count)",
    )


def check_cache_works(provider: Provider) -> CheckResult:
    """Anthropic: a repeated call sharing the long system prefix reports cache_read tokens.

    set_cache_control="auto" places the breakpoint at message index 1, so the
    cached prefix covers [system, user]. Both calls must therefore send the
    same [system, user] pair — only differing turns appended later would still
    hit the cache. We send the SAME pair twice, varying nothing.
    """
    cfg = LLMConfig(
        model_name=provider.model,
        temperature=1.0,
        max_completion_tokens=80,
        set_cache_control="auto",
    )
    llm = cfg.make()
    messages = [
        {"role": "system", "content": LONG_SYSTEM_PROMPT},
        {"role": "user", "content": "Reply with the single letter 'A'."},
    ]
    resp1 = llm(Prompt(messages=messages, tools=[]))
    resp2 = llm(Prompt(messages=messages, tools=[]))
    created = resp1.usage.cache_creation_tokens
    read = resp2.usage.cached_tokens
    if read > 0:
        return CheckResult(
            "prompt caching (Anthropic)",
            True,
            f"call1 cache_creation={created}, call2 cache_read={read}",
        )
    return CheckResult(
        "prompt caching (Anthropic)",
        False,
        f"call1 cache_creation={created}, call2 cache_read={read} (expected >0)",
    )


def check_tool_use_roundtrip(provider: Provider) -> CheckResult:
    """Tool-use loop with thinking: prior assistant turn's thinking_blocks round-trip.

    The integration form of TestPromptThinkingRoundTrip — call 1 emits thinking +
    a tool_call, call 2 echoes the assistant message back (with thinking_blocks)
    plus a tool result; if the signature is dropped or mangled, Anthropic returns
    a 400 here.
    """
    # Note: Anthropic rejects tool_choice="required" alongside reasoning_effort
    # ("Thinking may not be enabled when tool_choice forces tool use"), so we
    # stay on "auto" and shape the prompt to make the tool call obvious.
    cfg1 = LLMConfig(
        model_name=provider.model,
        temperature=1.0,
        # Headroom for reasoning_effort="low" (budget_tokens=1024 on Anthropic).
        max_completion_tokens=2048,
        reasoning_effort="low",
        tool_choice="auto",
    )
    llm1 = cfg1.make()
    user_msg = (
        "Use the `note` tool to record your final answer to this question, "
        f"then stop. Do not answer in plain text — only call the tool. "
        f"Question: {REASONING_QUESTION}"
    )
    resp1 = llm1(Prompt(messages=[{"role": "user", "content": user_msg}], tools=[NOTE_TOOL]))
    if not resp1.message.tool_calls:
        return CheckResult("tool-use round-trip", False, "no tool_call in call 1")
    tc = resp1.message.tool_calls[0]
    # Build follow-up: pass resp1.message (a litellm.Message) into Prompt; the
    # validator coerces it to a dict via model_dump(exclude_none=True), which
    # is the same path the harness uses in real episodes.
    prompt2 = Prompt(
        messages=[
            {"role": "user", "content": user_msg},
            resp1.message,
            {"role": "tool", "tool_call_id": tc.id, "content": "noted"},
            {"role": "user", "content": "In one short sentence, confirm what you noted."},
        ],
        tools=[NOTE_TOOL],
    )
    cfg2 = LLMConfig(
        model_name=provider.model,
        temperature=1.0,
        # Same headroom as call 1 — call 2 also has reasoning_effort enabled.
        max_completion_tokens=2048,
        reasoning_effort="low",
        tool_choice="auto",
    )
    llm2 = cfg2.make()
    try:
        llm2(prompt2)
    except Exception as e:
        return CheckResult(
            "tool-use round-trip",
            False,
            f"call 2 raised {type(e).__name__}: {str(e)[:160]}",
        )
    asst_dict = prompt2.messages[1]
    if not isinstance(asst_dict, dict):
        return CheckResult("tool-use round-trip", False, "assistant message not coerced to dict")
    if provider.name == "anthropic":
        blocks = asst_dict.get("thinking_blocks") or []
        if not blocks:
            return CheckResult(
                "tool-use round-trip",
                False,
                "assistant turn lost thinking_blocks after Prompt coercion (Anthropic would 400 here)",
            )
        sig = blocks[0].get("signature")
        if not sig:
            return CheckResult(
                "tool-use round-trip",
                False,
                "thinking_block present but signature missing — would break tool-use loop",
            )
        return CheckResult(
            "tool-use round-trip",
            True,
            f"thinking_blocks preserved with signature ({sig[:18]}…), call 2 accepted",
        )
    # OpenAI / Azure: no per-turn signature to track; success is the round-trip not raising.
    return CheckResult(
        "tool-use round-trip",
        True,
        "call 2 accepted (OpenAI does not require thinking_blocks echoed back)",
    )


def _cadence_two_turn(provider: Provider, *, reasoning_effort: str | None, interleaved: bool) -> tuple[int, int]:
    """One mode of the cadence probe — return (turn0_reasoning_tokens, turn1_reasoning_tokens).

    Pure tool-result continuation, mirroring how Genny actually drives a
    multi-step tool loop: turn 1's prompt ends with the ``tool`` message,
    with **no trailing user message**. That's the position where the
    interleaved-thinking gate engages — a new user message after the
    tool_result would let Anthropic reopen the turn (and emit thinking)
    regardless of the beta, defeating the probe.
    """
    common = {
        "model_name": provider.model,
        "temperature": 1.0,
        "max_completion_tokens": 2048,
        "tool_choice": "auto",
    }
    if reasoning_effort is not None:
        common["reasoning_effort"] = reasoning_effort
        common["interleaved_thinking"] = interleaved
    user_msg = (
        "Use the `note` tool to record a brief observation, then continue with a "
        f"one-sentence final answer to: {REASONING_QUESTION}"
    )
    cfg = LLMConfig(**common)
    resp1 = cfg.make()(Prompt(messages=[{"role": "user", "content": user_msg}], tools=[NOTE_TOOL]))
    t0 = resp1.usage.reasoning_tokens
    if not resp1.message.tool_calls:
        # Can't probe turn 1 without a tool_call to feed back; treat as ambiguous.
        return (t0, 0)
    tc = resp1.message.tool_calls[0]
    prompt2 = Prompt(
        messages=[
            {"role": "user", "content": user_msg},
            resp1.message,
            {"role": "tool", "tool_call_id": tc.id, "content": "noted"},
        ],
        tools=[NOTE_TOOL],
    )
    resp2 = cfg.make()(prompt2)
    return (t0, resp2.usage.reasoning_tokens)


def check_thinking_cadence(provider: Provider) -> CheckResult:
    """Perceive the three thinking cadences end-to-end: off / once / always.

    Each mode runs the same two-turn tool-use loop; we record reasoning_tokens
    for turn 0 and turn 1. The signature `(t0, t1)` distinguishes the modes:

      off    -> (0, 0)         — reasoning_effort=None
      once   -> (>0, 0)        — reasoning_effort set, interleaved_thinking=False (provider default)
      always -> (>0, >0)       — reasoning_effort set, interleaved_thinking=True (auto-fix(412) beta)

    Anthropic-only: OpenAI/Azure don't expose this cadence distinction
    (gpt-5 reasoning is server-managed), so the check is N/A there.
    """
    if provider.name != "anthropic":
        return CheckResult("thinking cadence (off/once/always)", True, "N/A — no cadence flag for this provider")

    off = _cadence_two_turn(provider, reasoning_effort=None, interleaved=False)
    once = _cadence_two_turn(provider, reasoning_effort="low", interleaved=False)
    always = _cadence_two_turn(provider, reasoning_effort="low", interleaved=True)
    summary = f"off={off} once={once} always={always}"

    if off != (0, 0):
        return CheckResult("thinking cadence (off/once/always)", False, f"off-mode emitted reasoning tokens: {summary}")
    if once[0] <= 0 or once[1] != 0:
        return CheckResult(
            "thinking cadence (off/once/always)",
            False,
            f"once-mode should be (>0, 0) — saw {once}. {summary}",
        )
    if always[0] <= 0 or always[1] <= 0:
        return CheckResult(
            "thinking cadence (off/once/always)",
            False,
            f"always-mode should be (>0, >0) — interleaved beta not engaging? {summary}",
        )
    return CheckResult("thinking cadence (off/once/always)", True, summary)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class ProviderResults:
    provider: Provider
    results: list[CheckResult] = field(default_factory=list)
    elapsed_s: float = 0.0


def _pass_fail(passed: bool) -> str:
    return (
        typer.style("PASS", fg=typer.colors.GREEN, bold=True)
        if passed
        else typer.style("FAIL", fg=typer.colors.RED, bold=True)
    )


def _emit(check: CheckResult) -> None:
    typer.echo(f"  {check.name:<38} {_pass_fail(check.passed)}  {check.details}")


def run_provider(provider: Provider) -> ProviderResults:
    out = ProviderResults(provider=provider)
    typer.echo(f"\n[{provider.name} / {provider.model}]")
    t0 = time.monotonic()
    captured: dict = {}
    for check in (
        check_no_reasoning(provider),
        check_reasoning_activates(provider, captured),
        check_reasoning_tokens(provider, captured),
        *([check_cache_works(provider)] if provider.supports_cache_test else []),
        check_tool_use_roundtrip(provider),
        check_thinking_cadence(provider),
    ):
        _emit(check)
        out.results.append(check)
    out.elapsed_s = time.monotonic() - t0
    typer.echo(f"  ({out.elapsed_s:.1f}s)")
    return out


def main(
    anthropic_model: Annotated[
        str,
        typer.Option(
            help=(
                "Anthropic model. Default is Sonnet 4.5: it has a 1024-token cache minimum so "
                "the cache check fits the smoke-test fixture. Haiku 4.5 also works for thinking "
                "and round-trip checks but has a higher cache-minimum that the fixture does not "
                "exceed, so the cache check will fail on it."
            )
        ),
    ] = "claude-sonnet-4-5",
    openai_model: Annotated[str, typer.Option(help="OpenAI model to test (direct, via OPENAI_API_KEY)")] = "gpt-5-mini",
    azure_model: Annotated[
        str | None,
        typer.Option(help="Azure-OpenAI deployment (e.g. azure/<deployment-name>) to test. Skipped if not set."),
    ] = None,
    skip_anthropic: Annotated[bool, typer.Option(help="Skip the Anthropic provider")] = False,
    skip_openai: Annotated[bool, typer.Option(help="Skip the direct-OpenAI provider")] = False,
) -> None:
    """Run the reasoning + caching + round-trip smoke tests against real APIs.

    Skips providers whose API keys are not set. Costs roughly $0.02-$0.05 per
    provider. Intended to be invoked by a human (or Claude Code on request),
    not by CI.
    """
    typer.echo(typer.style("=== Smoke test: reasoning + caching + tool-use round-trip ===", bold=True))

    all_results: list[CheckResult] = []

    # No-API gate
    v = check_validator()
    typer.echo("\n[validator (no API)]")
    _emit(v)
    all_results.append(v)

    # Provider matrix
    providers: list[Provider] = []
    if not skip_anthropic and os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(
            Provider(
                name="anthropic",
                model=anthropic_model,
                exposes_reasoning_text=True,
                supports_cache_test=True,
            )
        )
    elif not skip_anthropic:
        typer.echo(typer.style("\n[anthropic] SKIPPED — ANTHROPIC_API_KEY not set", fg=typer.colors.YELLOW))

    if not skip_openai and os.environ.get("OPENAI_API_KEY"):
        providers.append(
            Provider(
                name="openai",
                model=openai_model,
                exposes_reasoning_text=False,
                supports_cache_test=False,
            )
        )
    elif not skip_openai:
        typer.echo(typer.style("\n[openai] SKIPPED — OPENAI_API_KEY not set", fg=typer.colors.YELLOW))

    if azure_model:
        if os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"):
            providers.append(
                Provider(
                    name="azure",
                    model=azure_model,
                    exposes_reasoning_text=False,
                    supports_cache_test=False,
                )
            )
        else:
            typer.echo(
                typer.style(
                    f"\n[azure / {azure_model}] SKIPPED — AZURE_API_KEY (or AZURE_OPENAI_API_KEY) not set",
                    fg=typer.colors.YELLOW,
                )
            )

    if not providers:
        typer.echo("\nNo providers available — set ANTHROPIC_API_KEY / OPENAI_API_KEY / AZURE_API_KEY and retry.")
        # Cube-harness smoke contract: final line is `SMOKE OK|FAIL|SKIP: <name>`.
        # Exit 2 = SKIP. Even if the validator failed, we cannot exercise the
        # full behaviour without a provider, so SKIP is the correct outcome here.
        typer.echo("SMOKE SKIP: reasoning")
        raise typer.Exit(2)

    for p in providers:
        out = run_provider(p)
        all_results.extend(out.results)

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    typer.echo()
    typer.echo(f"Summary: {passed}/{total} checks passed.")
    if passed == total:
        typer.echo("SMOKE OK: reasoning")
        raise typer.Exit(0)
    typer.echo("SMOKE FAIL: reasoning")
    raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
