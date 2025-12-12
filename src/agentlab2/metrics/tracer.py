import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentlab2.metrics.disk_exporter import DiskSpanExporter
from agentlab2.metrics.processor import AL2_EXPERIMENT, AL2_NAME, AL2_TYPE, TYPE_EPISODE, TYPE_EXPERIMENT, TYPE_STEP


ENV_TRACEPARENT = "TRACEPARENT"
ENV_TRACE_OUTPUT = "AGENTLAB_TRACE_OUTPUT"
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"


class AgentTracer:
    """
    OTEL-native tracer for AgentLab2 experiments.

    Args:
        service_name: Name of the service/experiment for OTEL resource.
        output_dir: Directory to write trace files.
        otlp_endpoint: Optional OTLP endpoint for remote trace collection.
        set_global: If True (default), sets this as the global OTEL TracerProvider.
            This allows auto-instrumented libraries (LiteLLM, httpx, etc.) to emit
            spans into the same trace. Set to False for testing or multi-tracer scenarios.
    """


    def __init__(
            self,
            service_name: str,
            output_dir: str | Path,
            otlp_endpoint: str | None = None,
            set_global: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resource = Resource.create({SERVICE_NAME: service_name})
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(BatchSpanProcessor(DiskSpanExporter(self.output_dir)))

        if otlp_endpoint:
            os.environ[ENV_OTLP_ENDPOINT] = otlp_endpoint
        if os.environ.get(ENV_OTLP_ENDPOINT):
            self._provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

        if set_global:
            # Set as global provider so OTEL-instrumented libraries (LiteLLM, httpx, etc.)
            # emit spans into this trace
            trace.set_tracer_provider(self._provider)

        # Always use our provider directly (works with or without global)
        self._tracer = self._provider.get_tracer(__name__)
        self._current_experiment: str | None = None


    @contextmanager
    def benchmark(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute(AL2_TYPE, TYPE_EXPERIMENT)
            span.set_attribute(AL2_NAME, name)
            self._current_experiment = name
            _set_trace_env(self.output_dir)
            try:
                yield span
            finally:
                self._current_experiment = None


    @contextmanager
    def episode(
            self,
            name: str,
            experiment: str | None = None,
    ) -> Iterator[trace.Span]:
        exp = experiment or self._current_experiment or "default"
        parent_ctx = _get_parent_ctx_env()

        with self._tracer.start_as_current_span(name, context=parent_ctx) as span:
            span.set_attribute(AL2_TYPE, TYPE_EPISODE)
            span.set_attribute(AL2_NAME, name)
            span.set_attribute(AL2_EXPERIMENT, exp)
            yield span


    @contextmanager
    def step(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute(AL2_TYPE, TYPE_STEP)
            yield span


    def log(self, data: dict[str, Any], name: str = "step") -> None:
        with self.step(name) as span:
            for k, v in data.items():
                span.set_attribute(k, v)


    @contextmanager
    def span(self, name: str) -> Iterator[trace.Span]:
        with self._tracer.start_as_current_span(name) as span:
            yield span


    def shutdown(self) -> None:
        self._provider.shutdown()


def _set_trace_env(output_dir: Path) -> None:
    carrier: dict[str, str] = {}
    inject(carrier)
    if tp := carrier.get("traceparent"):
        os.environ[ENV_TRACEPARENT] = tp
    os.environ[ENV_TRACE_OUTPUT] = str(output_dir)


def _get_parent_ctx_env() -> Context | None:
    if tp := os.environ.get(ENV_TRACEPARENT):
        return extract({"traceparent": tp})
    return None


def get_trace_env_vars() -> dict[str, str]:
    # This helper is used only mainly by ray to propagate parameters to the workers.
    env_vars = {}
    if tp := os.environ.get(ENV_TRACEPARENT):
        env_vars[ENV_TRACEPARENT] = tp
    if output := os.environ.get(ENV_TRACE_OUTPUT):
        env_vars[ENV_TRACE_OUTPUT] = output
    if otlp := os.environ.get(ENV_OTLP_ENDPOINT):
        env_vars[ENV_OTLP_ENDPOINT] = otlp
    return env_vars


class _NoOpSpan:
    """A no-op span that ignores all attribute calls."""


    def set_attribute(self, key: str, value: Any) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


class _NoOpTracer:
    @contextmanager
    def episode(self, name: str, experiment: str | None = None) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN


    @contextmanager
    def step(self, name: str) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN


    @contextmanager
    def span(self, name: str) -> Iterator[_NoOpSpan]:
        yield _NOOP_SPAN


    def shutdown(self) -> None:
        pass


def get_tracer(service_name: str) -> AgentTracer | _NoOpTracer:
    output_dir = os.environ.get(ENV_TRACE_OUTPUT)
    if os.environ.get(ENV_TRACEPARENT) and output_dir:
        return AgentTracer(service_name=service_name, output_dir=output_dir)
    return _NoOpTracer()
