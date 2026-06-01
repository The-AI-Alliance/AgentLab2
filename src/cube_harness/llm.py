"""LLM interaction abstractions, LiteLLM based."""

import pprint
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Literal, Protocol
from uuid import uuid4

import litellm
import tenacity
from cube.core import TypedBaseModel, ValidatedConfig
from litellm import BadRequestError, Message, get_llm_provider
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.utils import token_counter
from pydantic import Field, field_validator, model_validator
from transformers import AutoTokenizer

# NOTE: Do not set litellm.callbacks = ["otel"] here at module level.
# When no TracerProvider is configured, litellm falls back to ConsoleSpanExporter
# which dumps huge JSON span dicts to stdout. Instead, enable the callback only
# after a proper TracerProvider has been set up (see metrics/tracer.py).

# Provider errors that will fail identically on retry — a typo'd model name, a bad
# API key, an unauthorized/oversized/policy-violating request. They are the
# complement of the transient set retried in `LLM._completion_with_retry`
# (5xx / 429 / timeouts / connection). `episode.py` maps these to the terminal,
# non-retriable INVALID_CONFIG status so the runner stops instead of burning the
# whole retry budget on a request that cannot succeed.
_PERMANENT_LLM_ERRORS: tuple[type[BaseException], ...] = (
    AuthenticationError,  # 401 — bad / missing key
    PermissionDeniedError,  # 403 — key lacks access to the model
    NotFoundError,  # 404 — model / endpoint does not exist (typo)
    BadRequestError,  # 400/422 — incl. ContextWindowExceeded, ContentPolicyViolation
)
_PERMANENT_HTTP_STATUS = frozenset({400, 401, 403, 404, 422})


def is_permanent_llm_error(exc: BaseException) -> bool:
    """True iff `exc` is an LLM provider error that will fail identically on retry.

    Classifies on the HTTP ``status_code`` first — provider-agnostic and set by the
    OpenAI-SDK base class that every litellm exception subclasses — then falls back
    to the exception type when no status is attached (e.g. connection errors, which
    are transient and correctly return False).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _PERMANENT_HTTP_STATUS
    return isinstance(exc, _PERMANENT_LLM_ERRORS)


class Prompt(TypedBaseModel):
    """Represents the input prompt to chat completion api of LLM."""

    messages: List[dict]
    tools: List[dict] = Field(default_factory=list)

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(cls, v: list) -> list[dict]:
        """Coerce LiteLLM Message objects to plain dicts.

        LiteLLM Message carries provider-specific fields (thinking_blocks,
        reasoning_content) that Pydantic doesn't know about, causing
        PydanticSerializationUnexpectedValue log spam on every model_dump call.
        """
        result: list[dict] = []
        for msg in v:
            if isinstance(msg, dict):
                result.append(msg)
            else:
                result.append(msg.model_dump(exclude_none=True))
        return result

    def __str__(self) -> str:
        """Debug view of the prompt."""
        messages = "\n".join([f"[{i}]{m}" for i, m in enumerate(self.messages)])
        tools = pprint.pformat(self.tools, width=120)
        return f"Tools:\n{tools}\nMessages[{len(self.messages)}]:\n{messages}"


class LLMConfig(ValidatedConfig):
    """Thin LLM wrapper around LiteLLM completion API."""

    model_name: str
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 1.0
    max_tokens: int = 128000
    max_model_len: int = 128000
    max_completion_tokens: int = 8192
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # Thinking cadence (Anthropic only — OpenAI/Azure gpt-5 reasoning is server-managed,
    # this flag is a no-op there). Combined with ``reasoning_effort`` you get three modes:
    #   off:    reasoning_effort=None                                   (no thinking)
    #   once:   reasoning_effort=<level>, interleaved_thinking=False    (think once at turn start; provider default)
    #   always: reasoning_effort=<level>, interleaved_thinking=True     (think after every tool result; needs the beta)
    # See auto-fix(412): in a multi-step tool-use loop (e.g. Genny swe), `once` means the
    # model thinks on step 0 and nowhere else — usually wrong for agents.
    interleaved_thinking: bool = False
    tool_choice: Literal["auto", "none", "required"] | None = "auto"
    parallel_tool_calls: bool = False
    logprobs: bool = False
    include_stop_str_in_output: bool | None = None
    skip_special_tokens: bool | None = None
    top_p: float | None = None
    top_k: int | None = None
    training: bool = False  # whether the call is for training (vs inference); may affect caching and logging behavior
    extra_body: dict = Field(default_factory=dict)
    num_retries: int = 5
    retry_strategy: Literal["exponential_backoff_retry", "constant_retry"] = "exponential_backoff_retry"
    timeout: float | None = 120.0  # seconds per attempt; None = no timeout
    # Anthropic prompt caching. "auto" places ephemeral cache_control breakpoints at the
    # system message and the last assistant message, plus the last tool definition. This
    # gives a stable anchor (system + tools) and a rolling boundary (last assistant) that
    # extends across steps as the conversation grows. No-op for non-Anthropic models.
    set_cache_control: Literal["auto"] | None = None

    @model_validator(mode="after")
    def _check_anthropic_thinking_temperature(self) -> "LLMConfig":
        """Anthropic extended thinking forbids temperature != 1.0; fail at config time, not API time."""
        if self.reasoning_effort is not None and _is_anthropic_model(self.model_name) and self.temperature != 1.0:
            raise ValueError(
                f"Anthropic extended thinking requires temperature=1.0, got temperature={self.temperature}. "
                "Either set temperature=1.0 or remove reasoning_effort."
            )
        return self

    # auto-fix(430)↓
    @model_validator(mode="after")
    def _check_interleaved_thinking_requires_reasoning(self) -> "LLMConfig":
        """``interleaved_thinking=True`` with ``reasoning_effort=None`` is a silent
        no-op (the "off" mode in the off/once/always table above): the beta header
        path is gated on ``reasoning_effort is not None`` in ``_complete``, so the
        flag is ignored and no thinking happens. Raise at config time so callers
        who think they enabled per-step thinking discover the miss before a
        million-token eval, not after.
        """
        if self.interleaved_thinking and self.reasoning_effort is None:
            raise ValueError(
                "interleaved_thinking=True is a no-op when reasoning_effort=None "
                "(this is the 'off' mode in the off/once/always table). Either set "
                "reasoning_effort to a level (to get 'always' mode) or drop "
                "interleaved_thinking. See LLMConfig docstring."
            )
        return self

    # /auto-fix(430)

    def make(self) -> "LLM":
        """Create LLM instance from config."""
        return LLM(config=self)

    def make_counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return partial(token_counter, model=self.model_name)


class Usage(TypedBaseModel):
    """Token usage information from LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # tokens read from cache (cache hit)
    cache_creation_tokens: int = 0  # tokens written to cache (Anthropic)
    # Reasoning/thinking tokens. LiteLLM surfaces these via
    # completion_tokens_details.reasoning_tokens for both OpenAI o-series/gpt-5
    # (native field) and Anthropic (normalized from thinking_blocks). They are
    # ALREADY counted within completion_tokens — do not add separately to a
    # budget tally or you will double-count.
    reasoning_tokens: int = 0
    cost: float = 0.0  # cost in USD from LiteLLM pricing


def get_reasoning(msg: Message) -> str:
    """Provider-agnostic reasoning text extractor — returns "" when no reasoning emitted.

    Checks reasoning_content (OpenAI o-series / gpt-5; Anthropic streaming) first,
    then concatenates thinking_blocks (Anthropic extended thinking). Returns the
    empty string when neither is present — it deliberately does NOT fall back to
    msg.content, since the final response text is already available on the
    Message and conflating it with thinking would muddy the contract.

    Works on any litellm.Message — including those reconstructed from persisted
    LLMCall.output records, making it the canonical reasoning extractor for both
    live runs and offline trajectory analysis.
    """
    if rc := getattr(msg, "reasoning_content", None):
        return rc
    blocks = getattr(msg, "thinking_blocks", None) or []
    return " ".join(b.get("thinking", "") for b in blocks if isinstance(b, dict))


class LLMResponse(TypedBaseModel):
    """Response from LLM containing message and usage info."""

    message: Message
    usage: Usage
    logprobs: list[float] | None = None
    completion_token_ids: list[int] | None = None
    finish_reason: str | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def reasoning_text(self) -> str:
        """Reasoning/thinking text emitted by the model, provider-agnostic. Empty when none."""
        return get_reasoning(self.message)


# auto-fix(412): Anthropic beta that lets Claude emit a thinking block
# after every tool result, not only on the first assistant turn. Required
# for per-step thinking in a multi-step tool-use agent (Genny swe mode).
_INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


def _is_anthropic_model(model_name: str) -> bool:
    """Does this model route to Anthropic's API (direct, Bedrock, or Vertex)?

    Uses LiteLLM's canonical provider resolver where possible so prefix-based
    routings are classified correctly — plain substring checks would false-positive
    on names like ``openai/something-claude-ish`` (resolver correctly returns
    ``provider=openai`` for that). Falls back to a substring check on the model
    name only when no routing prefix is present (e.g. brand-new ``claude-*`` names
    LiteLLM's registry hasn't caught up to). Used to gate Anthropic-specific
    payloads (cache_control) so they don't leak to other providers.
    """
    try:
        _, provider, _, _ = get_llm_provider(model_name)
    except BadRequestError:
        # Model not in LiteLLM's registry (e.g. ``claude-3-5-sonnet-20241022``,
        # ``claude-3-5-sonnet-latest``, or any new SKU that ships before LiteLLM
        # catches up). Fall back to substring matching, but only when no routing
        # prefix is present — keeps ``newprefix/claude-foo`` etc. from sneaking
        # through. Other exceptions propagate.
        if "/" in model_name:
            return False
        return "claude" in model_name.lower() or "anthropic" in model_name.lower()
    if provider == "anthropic":
        return True
    # Bedrock and Vertex route Claude models through the Anthropic API surface;
    # LiteLLM forwards cache_control for those routings.
    return provider in ("bedrock", "vertex_ai") and "claude" in model_name.lower()


def _msg_role(msg: Any) -> str | None:
    if isinstance(msg, dict):
        return msg.get("role")
    return getattr(msg, "role", None)


def _build_cache_injection_points(messages: list) -> list[dict]:
    """Return ephemeral cache_control breakpoints: second message + last assistant.

    Two breakpoints enable cross-step cache hits:

    1. Second message (index 1) — the goal / first large user content.  Marking
       this creates a stable seed cache (system + tools + goal) that all later
       steps can hit.  Marking only the system message fails because the system
       message is usually below Anthropic's 1 024-token minimum alone.

    2. Last assistant message — the rolling boundary.  On each new step the
       history grows by one (obs, asst) pair, so this breakpoint is always one
       message further out.  Anthropic's longest-prefix match hits the previous
       step's cache and writes a slightly longer entry.

    At step 0 (no assistant yet) only breakpoint 1 is emitted, writing the seed
    cache.  At step 1+, breakpoint 2 is also emitted; the lookup hits the seed
    (or the previous step's rolling cache) and the write extends it.
    """
    if len(messages) < 2:
        return []
    points: list[dict] = []
    control = {"type": "ephemeral"}
    # Breakpoint 1: second message — stable goal / main content anchor.
    points.append({"location": "message", "index": 1, "control": control})
    # Breakpoint 2: last assistant — rolling per-step extension.
    for i in range(len(messages) - 1, -1, -1):
        if _msg_role(messages[i]) == "assistant":
            if not any(p["index"] == i for p in points):
                points.append({"location": "message", "index": i, "control": control})
            break
    return points


def _mark_last_tool_for_cache(tools: list[dict]) -> list[dict]:
    """Return a copy of tools with ephemeral cache_control on the last entry.

    Caches the entire tools array prefix on Anthropic. LiteLLM passes the
    cache_control field through to the Anthropic API.
    """
    if not tools:
        return tools
    result = [dict(t) for t in tools]
    result[-1] = {**result[-1], "cache_control": {"type": "ephemeral"}}
    return result


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _completion_kwargs(self, prompt: Prompt) -> dict[str, Any]:
        tools = prompt.tools
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "tool_choice": self.config.tool_choice,
            "parallel_tool_calls": self.config.parallel_tool_calls,
            "messages": prompt.messages,
            "timeout": self.config.timeout,
        }
        if self.config.api_base is not None:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key
        if self.config.logprobs:
            kwargs["logprobs"] = True
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
            # auto-fix(412)↓ Anthropic only emits a thinking block AFTER a
            # tool result when the interleaved-thinking beta is set; without
            # it, a multi-step tool-use loop (Genny swe/flat_history) gets
            # thinking only on step 0. Gated by `interleaved_thinking` so
            # callers can pick: once-per-turn (provider default, cheaper) vs
            # every-step (this branch, deliberate). No-op for non-Anthropic.
            if self.config.interleaved_thinking and _is_anthropic_model(self.config.model_name):
                hdrs = dict(kwargs.get("extra_headers") or {})
                betas = [b for b in hdrs.get("anthropic-beta", "").split(",") if b.strip()]
                if _INTERLEAVED_THINKING_BETA not in betas:
                    betas.append(_INTERLEAVED_THINKING_BETA)
                hdrs["anthropic-beta"] = ",".join(betas)
                kwargs["extra_headers"] = hdrs
            # /auto-fix(412)
        if self.config.include_stop_str_in_output is not None:
            kwargs["include_stop_str_in_output"] = self.config.include_stop_str_in_output
        if self.config.skip_special_tokens is not None:
            kwargs["skip_special_tokens"] = self.config.skip_special_tokens
        if self.config.set_cache_control == "auto" and _is_anthropic_model(self.config.model_name):
            injection_points = _build_cache_injection_points(prompt.messages)
            if injection_points:
                kwargs["cache_control_injection_points"] = injection_points
            tools = _mark_last_tool_for_cache(tools)
        if tools:
            kwargs["tools"] = tools
        if not tools or self.config.tool_choice is None:
            # Drop tool_choice / parallel_tool_calls when there are no tools (some providers
            # reject tool_choice without a tools list) or when the caller opted out (None).
            kwargs.pop("tool_choice", None)
            kwargs.pop("parallel_tool_calls", None)
        return kwargs

    def __call__(self, prompt: Prompt) -> LLMResponse:
        kwargs = self._completion_kwargs(prompt)
        response = self._completion_with_retry(**kwargs)
        return self._response_from_completion(response)

    def _response_from_completion(self, response: Any) -> LLMResponse:
        usage = self._extract_usage(response)
        completion_logprobs = self._extract_completion_logprobs(response)
        return LLMResponse(
            message=response.choices[0].message,
            usage=usage,
            logprobs=[entry["logprob"] for entry in completion_logprobs] if completion_logprobs else None,
            completion_token_ids=[entry["token_id"] for entry in completion_logprobs] if completion_logprobs else None,
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )

    def _completion_with_retry(self, **kwargs: Any) -> Any:
        """Call litellm.completion with exponential backoff on transient errors.

        litellm's completion_with_retries caps its backoff at 10 s, which is too
        short for Anthropic overloaded_error responses under heavy load. We own the
        retry loop here to get a proper 120 s ceiling.
        """
        _RETRIABLE = (
            InternalServerError,
            ServiceUnavailableError,
            RateLimitError,
            Timeout,
            APIConnectionError,
        )
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential(multiplier=2, max=120),
            stop=tenacity.stop_after_attempt(self.config.num_retries),
            retry=tenacity.retry_if_exception_type(_RETRIABLE),
            reraise=True,
        )
        return retryer(litellm.completion, **kwargs)

    def _extract_usage(self, response) -> Usage:
        """Extract usage info from LiteLLM response."""
        usage_data = getattr(response, "usage", None)
        if usage_data is None:
            return Usage()

        def safe_int(value: object) -> int:
            """Safely convert a value to int, returning 0 for non-numeric types."""
            if isinstance(value, int):
                return value
            return 0

        def safe_float(value: object) -> float:
            """Safely convert a value to float, returning 0.0 for non-numeric types."""
            if isinstance(value, (int, float)):
                return float(value)
            return 0.0

        cached_tokens = 0
        cache_creation_tokens = 0

        # Check prompt_tokens_details for cached_tokens (OpenAI/Anthropic)
        prompt_details = getattr(usage_data, "prompt_tokens_details", None)
        if prompt_details:
            cached_tokens = safe_int(getattr(prompt_details, "cached_tokens", 0))

        # Anthropic-specific fields
        cache_creation_tokens = safe_int(getattr(usage_data, "cache_creation_input_tokens", 0))
        cache_read = safe_int(getattr(usage_data, "cache_read_input_tokens", 0))
        if cache_read > 0:
            cached_tokens = cache_read  # Anthropic uses this field name

        # Extract cost from LiteLLM's hidden params
        cost = 0.0
        hidden_params = getattr(response, "_hidden_params", {})
        if isinstance(hidden_params, dict):
            cost = safe_float(hidden_params.get("response_cost", 0.0))

        # Reasoning tokens — LiteLLM normalizes both OpenAI (native field) and
        # Anthropic (computed from thinking_blocks) into completion_tokens_details.
        # These are already part of completion_tokens; the separate field is for
        # telemetry, not for budgeting.
        reasoning_tokens = 0
        completion_details = getattr(usage_data, "completion_tokens_details", None)
        if completion_details:
            reasoning_tokens = safe_int(getattr(completion_details, "reasoning_tokens", 0))

        return Usage(
            prompt_tokens=safe_int(getattr(usage_data, "prompt_tokens", 0)),
            completion_tokens=safe_int(getattr(usage_data, "completion_tokens", 0)),
            total_tokens=safe_int(getattr(usage_data, "total_tokens", 0)),
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=cost,
        )

    def _extract_completion_logprobs(self, response) -> list[dict[str, int | float]]:
        """Extract completion logprobs and token IDs from an OpenAI-compatible response."""
        result: list[dict[str, int | float]] = []
        choice = response.choices[0]
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None:
            return result
        content = getattr(logprobs, "content", None)
        if content is None:
            return result
        for entry in content:
            token_str = getattr(entry, "token", None)
            token_id: int | None = None
            if isinstance(token_str, str) and token_str.startswith("token_id:"):
                try:
                    token_id = int(token_str.split(":", 1)[1])
                except ValueError:
                    token_id = None
            logprob = getattr(entry, "logprob", None)
            if token_id is None or not isinstance(logprob, (int, float)):
                continue
            result.append({"token_id": token_id, "logprob": float(logprob)})
        return result

@dataclass(frozen=True)
class LLMRouteLease:
    """A temporary route assignment for one generation request."""

    route_id: str
    api_base: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    metadata: dict | None = None


class LLMRouter(Protocol):
    """Routes individual LLM calls for `RoutedLLM`."""

    def acquire(self, config: LLMConfig, prompt: Prompt) -> LLMRouteLease: ...

    def release(
        self,
        lease: LLMRouteLease,
        response: LLMResponse | None = None,
        error: BaseException | None = None,
    ) -> None: ...

class DummyRouter(LLMRouter):
    """A dummy router that performs no routing and returns an empty lease."""

    def acquire(self, config: LLMConfig, prompt: Prompt) -> LLMRouteLease:
        return LLMRouteLease(route_id="dummy")

    def release(
        self,
        lease: LLMRouteLease,
        response: LLMResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        pass

class VLLMTokenCounter:
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

    def count_prompt_tokens(self, messages, tools=None) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_special_tokens=True,
            add_generation_prompt=True,
            tokenize=True,
        )
        return len(token_ids)


class RoutedLLMConfig(LLMConfig):
    """LLM config variant that routes each generation through a router.

    The router is intentionally excluded from serialization so existing episode
    configs and result artifacts stay portable.
    """

    tokenizer_name: str # used for token counting; can differ from model_name in LLMConfig when routing to different models
    router: Any = Field(default=None, exclude=True)

    def make(self) -> "RoutedLLM":
        return RoutedLLM(config=self)

    def make_counter(self) -> Callable[..., int]:
        """Get a token counter function for the LLM model."""
        return VLLMTokenCounter(tokenizer_name=self.tokenizer_name).count_prompt_tokens


class RoutedLLM(LLM):
    """LLM wrapper with per-request routing and admission control."""

    config: RoutedLLMConfig

    def __init__(self, config: RoutedLLMConfig):
        super().__init__(config=config)

    def __call__(self, prompt: Prompt) -> LLMResponse:
        if self.config.router is None:
            return super().__call__(prompt)

        lease: LLMRouteLease | None = None
        response_obj: LLMResponse | None = None
        error: BaseException | None = None
        started_at: datetime | None = None
        started_perf: float | None = None
        try:
            lease = self.config.router.acquire(self.config, prompt)
            kwargs = self._completion_kwargs(prompt)
            if lease.api_base is not None:
                kwargs["api_base"] = lease.api_base
            if lease.api_key is not None:
                kwargs["api_key"] = lease.api_key
            if lease.model_name is not None:
                kwargs["model"] = lease.model_name

            started_at = datetime.now()
            started_perf = time.perf_counter()
            raw_response = self._completion_with_retry(**kwargs)
            response_obj = self._response_from_completion(raw_response)
            finished_at = datetime.now()
            response_obj.metadata.update(lease.metadata or {})
            response_obj.metadata.update(
                {
                    "route_id": lease.route_id,
                    "route_api_base": lease.api_base,
                    "route_model_name": lease.model_name,
                    "llm_started_at": started_at.isoformat(),
                    "llm_finished_at": finished_at.isoformat(),
                    "llm_latency_s": time.perf_counter() - started_perf if started_perf is not None else None,
                }
            )
            return response_obj
        except BaseException as exc:
            error = exc
            raise
        finally:
            if lease is not None:
                self.config.router.release(lease, response=response_obj, error=error)


class LLMCall(TypedBaseModel):
    """Represents a call to an LLM model."""

    id: str = Field(default_factory=lambda: uuid4().hex)  # unique storage key
    tag: str = ""  # optional label shown as tab name in viewers (e.g. "act", "summary")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_config: LLMConfig
    prompt: Prompt
    prompt_tokens: int = -1  # Number of tokens in the prompt; set to -1 if unknown or not applicable
    output: Message
    output_tokens: int = -1  # Number of tokens in the output; set to -1 if unknown or not applicable
    usage: Usage = Field(default_factory=Usage)
    logprobs: list[float] | None = None
    completion_token_ids: list[int] | None = None
    finish_reason: str | None = None
    metadata: dict = Field(default_factory=dict)


# === auto-fix notes ===
# auto-fix-note(412) {class=L1 issue=412 hash=PENDING ctx=anthropic/claude-haiku-4-5/genny-swe/cube-harness@5ca4e565}
#   symptoms:  Genny swe/flat_history + claude-haiku-4-5 + reasoning_effort
#              on terminalbench2: extended thinking fired ONLY on step 0
#              (step0=145 reasoning tokens, steps 1-14 = exactly 0,
#              thinking_blocks=0). 15-step probe: 1/15 steps thought.
#   invariant: a reasoning-enabled multi-step tool-use agent must be able
#              to think on every step, not only the first assistant turn.
#   why:       Anthropic only emits thinking after a tool result when the
#              interleaved-thinking beta is set; llm.py passed
#              reasoning_effort but not the beta. Exposed as an opt-in
#              `LLMConfig.interleaved_thinking` flag (default False = match
#              provider default ("once"); set True = "always"), so callers
#              can pick a deliberate cadence instead of being silently
#              pinned to either one. Right layer = the LLM wrapper that
#              owns provider params. No contract change -> L1.
#   tested:    tests/test_llm.py::TestInterleavedThinkingBeta (beta gated
#              on the flag: present iff anthropic + reasoning_effort +
#              interleaved_thinking; absent otherwise) + scripts/smoke/
#              reasoning.py adds an off/once/always cadence probe (live
#              Anthropic, asserts per-turn reasoning_token pattern) +
#              original validation probe: 15/15 steps think with the
#              flag on, 255 -> 846 reasoning tokens.
# auto-fix-note(430) {class=L1 anchor=PR#430 hash=f513c550 ctx=anthropic/cube-harness/genny-swe/silent-no-op}
