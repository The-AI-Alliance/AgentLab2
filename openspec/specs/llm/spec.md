# LLM Integration

**Module:** `cube_harness.llm`

## Purpose

Thin wrapper over [LiteLLM](https://docs.litellm.ai/) that standardizes prompt
construction, retry behavior, and usage accounting. All LLM calls in the harness
flow through this module — per the constitution, direct SDK use (OpenAI SDK,
Anthropic SDK) is forbidden (PS-002).

## Public API

### `LLMConfig`
```python
class LLMConfig(TypedBaseModel):
    model_name: str
    temperature: float = 1.0
    max_tokens: int = 128000
    max_completion_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    tool_choice: Literal["auto", "none", "required"] | None = "auto"   # None opts out
    parallel_tool_calls: bool = False
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"
    timeout: float | None = 120.0       # seconds per attempt; None disables
    set_cache_control: Literal["auto"] | None = None   # Anthropic prompt caching, see "Caching"

    def make(self) -> LLM
    def make_counter(self) -> Callable[..., int]   # partial(token_counter, model=model_name)
```

### `Prompt`
```python
class Prompt(TypedBaseModel):
    messages: list[dict]              # litellm.Message inputs are coerced via a
                                      # field_validator (model_dump(exclude_none=True))
    tools: list[dict] = []
```

Callers may pass a mix of `dict` and `litellm.Message` objects — the validator
normalises to plain dicts at construction. This keeps serialisation noise-free
(Message's dynamic provider-specific fields like `thinking_blocks`,
`reasoning_content` would otherwise trip `PydanticSerializationUnexpectedValue`
on every `model_dump`) and gives downstream readers a single homogenous type
to work with.

### `LLMResponse` / `Usage`
```python
class LLMResponse(TypedBaseModel):
    message: Message          # litellm.Message
    usage: Usage

    @property
    def reasoning_text(self) -> str   # provider-agnostic; empty when no reasoning

class Usage(TypedBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0    # Anthropic prompt caching
    reasoning_tokens: int = 0          # LiteLLM-normalized across providers; subset of completion_tokens
    cost: float = 0.0                  # USD from LiteLLM pricing
```

### `get_reasoning(msg: Message) -> str`

Module-level helper. Provider-agnostic reasoning extractor:

1. `msg.reasoning_content` if non-empty (OpenAI / streaming).
2. Concatenation of `msg.thinking_blocks[*].thinking` (Anthropic extended thinking).
3. Fallback to `msg.content`, else empty string.

Works on any `litellm.Message`, including those reconstructed from persisted
`LLMCall.output` records — so it's the canonical reasoning extractor for both
live runs (`LLMResponse.reasoning_text`) and offline trajectory analysis.

### `LLM`
```python
class LLM:
    def __init__(self, config: LLMConfig)
    def __call__(self, prompt: Prompt) -> LLMResponse
    # Uses litellm.completion_with_retries under the hood with config.retry_strategy.
```

### `LLMCall` (logged record)
```python
class LLMCall(TypedBaseModel):
    id: str = field(default_factory=lambda: str(uuid4()))
    tag: str | None = None           # e.g. "act", "summary", "criticise"
    timestamp: datetime
    config: LLMConfig
    prompt: Prompt
    output: Message
    usage: Usage | None = None
```

Captured in `AgentOutput.llm_calls`. Agents MUST set `tag` to distinguish multi-call
steps in traces and training data.

## Invariants

1. All LLM calls route through `LLM.__call__` — no direct use of `litellm.completion`
   in the harness code.
2. Retry strategy is determined by `LLMConfig`, not the call site.
3. `LLMCall.tag` is the primary way to correlate multiple LLM calls in one agent step.
4. Module-level `litellm.callbacks` is intentionally NOT set. OTel callbacks are
   attached only after a proper `TracerProvider` is configured (see metrics spec) —
   otherwise litellm's default console exporter floods stdout.
5. **Reasoning round-trip.** `Prompt._coerce_messages` MUST preserve provider
   reasoning fields on assistant messages so they can be re-sent in subsequent
   calls. Specifically: `thinking_blocks` (including each block's `signature`),
   `reasoning_content`, and `tool_calls` survive coercion from `litellm.Message`
   to dict. Anthropic extended thinking with tool use requires the prior turn's
   `thinking_blocks` to be echoed back; stripping them breaks the tool-use loop.
6. **Anthropic thinking + temperature.** `LLMConfig` rejects construction when
   `reasoning_effort` is set on an Anthropic model with `temperature != 1.0`.
   Anthropic forbids non-unit temperature under extended thinking; the validator
   surfaces this at config time rather than at API time.

## Caching (Anthropic)

When `LLMConfig.set_cache_control == "auto"` and the configured model routes to
Anthropic (direct, Bedrock, or Vertex — detected via `litellm.get_llm_provider`
with a substring fallback for model names LiteLLM's registry hasn't catalogued
yet), `LLM.__call__` places ephemeral `cache_control` breakpoints at:

1. **Message index 1** — the goal / first large user observation. Stable anchor
   that lifts the cached prefix above Anthropic's 1024-token minimum (the system
   message alone is typically under that floor).
2. **Last assistant message** — rolling per-step boundary. Each new step extends
   the cached prefix by one (obs, asst) pair via Anthropic's longest-prefix
   match.
3. **Last tool definition** — caches the entire tools array prefix.

Breakpoint injection is done via LiteLLM's `cache_control_injection_points`
hook (canonical public API; LiteLLM handles the wire-format reshape into
Anthropic's content-block-with-cache_control structure). For non-Anthropic
models the flag is a no-op — the payload is never emitted.

`Usage.cached_tokens` / `Usage.cache_creation_tokens` are populated from the
Anthropic response so trace consumers can see cache-hit rates per step.

## Contracts for implementers

- Agent implementations build a `Prompt` and call `self.llm(prompt)`. Record the
  call:
  ```python
  call = LLMCall(tag="act", config=self.config.llm_config, prompt=prompt,
                 output=resp.message, usage=resp.usage)
  output.llm_calls.append(call)
  ```
- For multi-model agents, use one `LLM` per model — the class holds a single config.
- Pass a token counter from `config.make_counter()` for prompt-size budgeting.

## Gotchas

- `completion_with_retries` returns on first success, but retries count toward the
  per-attempt timeout. Total call time is bounded by `num_retries * timeout` in the
  worst case.
- `Prompt.messages` accepts both dicts and `litellm.Message` objects; the
  `field_validator` coerces Messages to dicts at construction so the stored type
  is always `list[dict]`. Downstream readers don't need to handle the union.
- **Reasoning extraction.** Set `reasoning_effort` to activate native reasoning on
  supported models (OpenAI o-series / gpt-5; Anthropic Claude 3.7+/4.x; Gemini 2.5;
  Grok 3/4; DeepSeek R1/R2; Qwen3-thinking; Magistral). Use
  `response.reasoning_text` (or `get_reasoning(msg)` for offline analysis) to obtain
  the thinking string for `AgentOutput.thoughts`. The structured form is preserved
  on `response.message.thinking_blocks` / `reasoning_content` for round-trip.
- **OpenAI hides the reasoning text.** OpenAI o-series and gpt-5 (including
  Azure-OpenAI deployments) return `reasoning_tokens > 0` to confirm the model
  thought, but **do not return the thinking text** — `reasoning_content` and
  `thinking_blocks` are empty even with `reasoning_effort` set. This is an
  OpenAI design choice. Consequence: `AgentOutput.thoughts` will be `None` on
  OpenAI episodes even when the model reasoned. The agent still benefits from
  the reasoning; only the human-readable trace is unavailable.
- **Anthropic thinking constraints.** Two API-level restrictions surface when
  `reasoning_effort` is set on an Anthropic model:
  1. `temperature` must be `1.0`. `LLMConfig` validates this at construction time.
  2. `tool_choice` must NOT be `"required"`. Anthropic returns 400 with
     "Thinking may not be enabled when tool_choice forces tool use." Stay on
     `"auto"` (the default) and shape the prompt to elicit the tool call.
  3. `max_completion_tokens` must exceed the thinking `budget_tokens` LiteLLM
     maps `reasoning_effort` onto (≈1024 for "low", more for higher). Set
     `max_completion_tokens` to at least 2048 when reasoning is active.
- **Tool-use loops with Anthropic thinking.** Each assistant turn's
  `thinking_blocks` (including `signature`) MUST be echoed back in subsequent
  calls. `Prompt._coerce_messages` preserves them automatically via
  `Message.model_dump(exclude_none=True)`. Do not strip these fields.
- **`reasoning_tokens` accounting.** LiteLLM normalizes `reasoning_tokens` for
  both OpenAI (native `completion_tokens_details.reasoning_tokens`) and
  Anthropic (computed from `thinking_blocks`) into the same `Usage` field.
  These tokens are **already counted inside `completion_tokens`** — do not add
  them to a budget tally, or you will double-count. The field exists for
  telemetry only.
- Cost is USD from LiteLLM's built-in pricing — may lag behind provider price changes.
