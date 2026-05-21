# Metrics & Telemetry

**Module:** `cube_harness.metrics`

## Purpose

OpenTelemetry-based tracing for the harness. Benchmark → Episode → Step span hierarchy
with tool_span children. Exports to an OTLP collector (Phoenix, Jaeger, Honeycomb, …).

All tracing is opt-in: if OTel dependencies are missing or no `TracerProvider` is
configured, a no-op tracer is returned and the harness runs unchanged.

## Public API

### `get_tracer(exp_name, otlp_endpoint=None, model=None, agent_name=None) -> Tracer`
Factory. Returns either `_AgentTracer` (when OTLP is configured) or a no-op stub.
Sets `SERVICE_NAME = exp_name`. If `otlp_endpoint` is provided, configures the
OTLP HTTP exporter.

### `Tracer` protocol
```python
class Tracer(Protocol):
    @contextmanager
    def benchmark(self, name: str) -> Iterator[trace.Span]
    @contextmanager
    def episode(self, name: str, experiment: str | None = None) -> Iterator[trace.Span]
    @contextmanager
    def step(self, name: str) -> Iterator[trace.Span]
    @contextmanager
    def span(self, name: str) -> Iterator[trace.Span]
    def log(self, data: dict[str, Any], name: str = "step") -> None
    def shutdown(self) -> None
```

### `tool_span(action: Action) -> Iterator[Span]`
Module-level context manager that tool dispatchers wrap around `execute_action`. Records:
- `gen_ai.tool.name` = `action.name`
- `gen_ai.tool.call.id` = `action.id` (or `""`)
- `gen_ai.tool.call.arguments` = JSON-encoded `action.arguments`

(`gen_ai.tool.call.result` is set by the tool wrapper after execution.)

### Ray propagation
Trace context travels to Ray workers via env vars:
- `TRACEPARENT` — W3C trace context
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
- `CUBE_HARNESS_MODEL`
- `CUBE_HARNESS_AGENT_NAME`

`get_trace_env_vars()` returns the dict to pass in `ray.init(runtime_env={"env_vars": ...})`.

## Span attributes (semantic conventions)

Following the [GenAI OTel semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

| Attribute | Where set |
|-----------|-----------|
| `ch.type` = experiment\|episode\|step | On each span level |
| `ch.name` | Human-readable name |
| `ch.experiment` | Episode and step spans |
| `ch.episode` | Propagated via baggage to nested spans |
| `gen_ai.request.model` | Episode (from `CUBE_HARNESS_MODEL`) |
| `gen_ai.agent.name/id/description` | Episode |
| `gen_ai.tool.name/call.id/call.arguments/call.result` | tool_span |

`_EpisodeBaggageSpanProcessor` auto-propagates `ch.episode` to all descendant spans
so LLM-call spans created by litellm inherit the episode context.

## Invariants

1. Module import does NOT mutate global state (no `litellm.callbacks = [...]` at
   import time). The callback is set up only after a real `TracerProvider` is
   configured — avoids litellm's ConsoleSpanExporter flooding stdout.
2. `get_tracer()` always returns a usable object — if OTel fails to import, callers
   get a silent no-op, not an exception.
3. Step spans live inside episode spans; episode spans live inside benchmark spans.
   Violating the hierarchy breaks downstream analysis tools.
4. Ray workers that don't receive `TRACEPARENT` start a new trace root. Passing env
   vars via `ray.init(runtime_env=...)` is mandatory to preserve the tree.

## Contracts for implementers

- Wrap new long-running operations in `tracer.span(name)` so they appear in the
  trace hierarchy.
- Tool dispatchers should wrap `execute_action` in `tool_span(action)` to surface
  invocations in the trace.
- On shutdown, always call `tracer.shutdown()` — BatchSpanProcessor flushes pending
  spans to the exporter. The runners already do this in `finally` blocks.

## Gotchas

- OTLP export is HTTP-based — if the endpoint is unreachable, spans are dropped
  silently (BatchSpanProcessor retries then discards). Runs don't fail.
- The GenAI semantic conventions are still draft — attribute names may change
  across OTel versions.
- Ray's dashboard auto-collects telemetry; don't conflate Ray dashboard metrics
  with OTLP traces. The OTLP endpoint is the authoritative source for ADP-style
  analysis.
