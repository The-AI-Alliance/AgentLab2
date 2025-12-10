from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agentlab2.metrics.processor import AL2_TYPE, TYPE_EPISODE, TraceProcessor
from agentlab2.metrics.store import JsonlSpanWriter


class DiskSpanExporter(SpanExporter):
    def __init__(self, run_dir: str) -> None:
        self._store = JsonlSpanWriter(run_dir)
        self._processor = TraceProcessor(run_dir)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self._store.write_span(span)

        for span in spans:
            if dict(span.attributes or {}).get(AL2_TYPE) == TYPE_EPISODE:
                self._processor.export_episode(span)

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._store.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._store.flush()
