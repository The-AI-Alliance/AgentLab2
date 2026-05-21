# RFC: Reasoning / Extended Thinking as a First-Class LLM Concern

**Status:** DRAFT
**Author:** Alexandre Lacoste
**Date:** 2026-05-13

---

## Problem

Modern reasoning models (OpenAI o-series and gpt-5 family, Anthropic Claude 3.7+/4.x,
Gemini 2.5, Grok 3/4, DeepSeek R1/R2, Qwen3-thinking, Magistral) emit a structured
*thinking* artifact alongside their final response — `thinking_blocks` (Anthropic),
`reasoning_content` (OpenAI), or equivalent. LiteLLM normalizes these onto the
`Message` object that flows through `cube_harness.llm`.

cube-harness treats this artifact ad-hoc:

1. **Per-agent extraction.** Only Genny2 knows how to pull the reasoning text out
   of a response (private `_get_reasoning`, called from three sites). `react.py`,
   `genny.py`, and any future agent have to reinvent the same provider-fanout
   logic to populate `AgentOutput.thoughts`.

2. **No usage accounting.** Reasoning tokens are billable but invisible. OpenAI
   exposes them as `completion_tokens_details.reasoning_tokens`; we drop the field
   in `_extract_usage`. Cost dashboards therefore can't separate "tokens spent
   thinking" from "tokens spent answering."

3. **No round-trip verification.** Anthropic extended thinking requires that
   `thinking_blocks` (with `signature`) be echoed back in subsequent assistant
   turns when tool use is in the loop. The current `Prompt._coerce_messages` does
   preserve them (verified empirically: `Message.model_dump(exclude_none=True)`
   keeps the field), but there is no test locking this in. A future refactor
   could silently break tool-use loops on Claude with thinking enabled.

4. **No validator for Anthropic's `temperature=1` constraint.** Setting
   `reasoning_effort` on a Claude model with `temperature != 1` produces a 400
   from the provider. Today this fails at run time; it should fail at config
   time.

---

## Scope

A small refactor of `cube_harness.llm` that promotes reasoning to a first-class
LLM concern, so every agent gets it uniformly:

- `Usage.reasoning_tokens` — extracted automatically per call.
- `LLMResponse.reasoning_text` — provider-agnostic accessor for the thinking text.
- Module-level `get_reasoning(msg) -> str` helper for offline analysis
  (`LLMCall.output`, persisted trajectories).
- `LLMConfig` validator: Anthropic + `reasoning_effort` requires `temperature == 1`.
- Regression test for the `Prompt` round-trip: `thinking_blocks` + `signature`
  survive coercion and are accepted back by LiteLLM.

Genny2 stops being the special case: its private `_get_reasoning` is removed
and all three call sites are replaced with calls to the shared helper.

**Out of scope:**

- New configuration knobs for explicit thinking budget control
  (`reasoning_effort` is sufficient; LiteLLM does the budget mapping).
- Changes to `AgentOutput.thoughts` (stays `str | None` for trajectory back-compat).
- Special XRay rendering of structured thinking blocks (flat text suffices).
- Changes to cube-standard.

---

## Design

### Layer 1 — `Usage` (new field)

```python
class Usage(TypedBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0   # NEW
    cost: float = 0.0
```

Extracted in `_extract_usage` from
`usage.completion_tokens_details.reasoning_tokens`. LiteLLM normalizes both
OpenAI (native field) and Anthropic (computed from `thinking_blocks`) into this
path, so the extraction is provider-agnostic. The value is **already part of
`completion_tokens`** — telemetry only, not budgeting.

Default of 0 means existing trajectory JSONs without the field deserialize cleanly.

### Layer 2 — `LLMResponse.reasoning_text` + `get_reasoning(msg)`

Promote Genny2's private `_get_reasoning` to `cube_harness.llm`:

```python
def get_reasoning(msg: "Message") -> str:
    """Provider-agnostic reasoning text extractor.

    Checks reasoning_content (OpenAI / Anthropic streaming),
    then thinking_blocks (Anthropic extended thinking),
    then falls back to plain content.
    """
    if rc := getattr(msg, "reasoning_content", None):
        return rc
    blocks = getattr(msg, "thinking_blocks", None) or []
    text = " ".join(b.get("thinking", "") for b in blocks if isinstance(b, dict))
    if text:
        return text
    return msg.content or ""


class LLMResponse(TypedBaseModel):
    message: Message
    usage: Usage

    @property
    def reasoning_text(self) -> str:
        return get_reasoning(self.message)
```

Call-site impact in Genny2: three sites previously calling private
`_get_reasoning(...)` now call `get_reasoning(...)` from `cube_harness.llm`. Any
future agent gets thinking with the same one-liner.

### Layer 3 — `LLMConfig` validator

```python
@model_validator(mode="after")
def _check_anthropic_thinking_temperature(self) -> "LLMConfig":
    if (
        self.reasoning_effort is not None
        and _is_anthropic_model(self.model_name)
        and self.temperature != 1.0
    ):
        raise ValueError(
            f"Anthropic extended thinking requires temperature=1.0, "
            f"got temperature={self.temperature}. "
            f"Either set temperature=1.0 or remove reasoning_effort."
        )
    return self
```

Fail at config construction, not at API call time.

### Layer 4 — Round-trip regression test

```python
def test_thinking_blocks_survive_prompt_coercion() -> None:
    msg = Message(
        role="assistant",
        content="here is my answer",
        thinking_blocks=[
            {"type": "thinking", "thinking": "let me think...", "signature": "sig_abc123"}
        ],
        reasoning_content="let me think...",
    )
    prompt = Prompt(messages=[msg], tools=[])
    coerced = prompt.messages[0]

    assert coerced["thinking_blocks"][0]["signature"] == "sig_abc123"
    assert coerced["thinking_blocks"][0]["thinking"] == "let me think..."
    assert coerced["reasoning_content"] == "let me think..."
```

No live API call. Locks in the current empirical behavior as a contract.

### Layer 5 — Optional: OTel span attribute

LiteLLM's OTel callback emits `gen_ai.usage.*` attributes automatically. If
`reasoning_tokens` is exposed in `Usage`, we should additionally emit
`gen_ai.usage.reasoning_tokens` on the LLM client span — matching the GenAI
semantic convention. This is a one-line addition in whichever code path
populates the harness-side LLM span (or a config tweak on the LiteLLM OTel
callback if it does not emit it by default).

Verification deferred to implementation: if LiteLLM already emits this
attribute, no harness change is needed.

---

## Migration

Call sites to update:

- `genny2.py`: replace three `_get_reasoning(...)` calls with `get_reasoning(...)`
  imported from `cube_harness.llm`; delete the private helper.

No serialization break: `Usage.reasoning_tokens` defaults to 0 for old records.
No API break: `LLMResponse.reasoning_text` is additive; existing
`response.message.thinking_blocks` access still works.

---

## Alternatives considered

**Add a `thinking_budget_tokens` knob to `LLMConfig`.** Rejected for V1:
`reasoning_effort` (low/medium/high) is the unified abstraction LiteLLM provides,
and over-parameterization here would push provider-specific concerns into the
config. Easy to add later if a need emerges.

**Make `AgentOutput.thoughts` a structured type (list of thinking blocks).**
Rejected: changes the trajectory storage format and breaks XRay rendering for
zero capability gain. The structured form is already preserved inside
`LLMCall.output`; tooling that wants it can read it from there.

**Auto-override temperature instead of validating.** Rejected: silent overrides
of user-supplied config violate the constitution's explicitness pillar. Better
to fail loudly with a clear message.

**Move the helper to cube-standard.** Rejected: reasoning is an LLM-layer
artifact, not a Task/Tool/Benchmark contract. cube-standard has no opinion on
how agents introspect LLM responses.

---

## Open questions

1. Does LiteLLM already emit `gen_ai.usage.reasoning_tokens` on its OTel client
   spans? If yes, harness change is zero for Layer 5. If no, where is the cleanest
   place to add it.
2. `Usage.reasoning_tokens` is for telemetry only — for OpenAI o-series the
   reasoning tokens are already counted inside `completion_tokens`, so adding
   them to a budget tally would double-count. Documented in spec gotchas.
