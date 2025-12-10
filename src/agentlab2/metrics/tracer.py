import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agentlab2.metrics.disk_exporter import DiskSpanExporter
from agentlab2.metrics.processor import AL2_EXPERIMENT, AL2_NAME, AL2_TYPE, TYPE_EPISODE, TYPE_EXPERIMENT, TYPE_STEP


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
        output_dir: str,
        otlp_endpoint: str | None = None,
        set_global: bool = True,
    ) -> None:
        self.run_id = str(uuid.uuid4())
        self.output_dir = Path(output_dir) / "metrics" / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resource = Resource.create({SERVICE_NAME: service_name})
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(BatchSpanProcessor(DiskSpanExporter(str(self.output_dir))))

        if otlp_endpoint:
            self._provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

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
            try:
                yield span
            finally:
                self._current_experiment = None

    @contextmanager
    def episode(self, name: str, experiment: str | None = None) -> Iterator[trace.Span]:
        exp = experiment or self._current_experiment or "default"
        with self._tracer.start_as_current_span(name) as span:
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
