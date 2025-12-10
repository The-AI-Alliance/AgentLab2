import os
import threading

from opentelemetry.sdk.trace import ReadableSpan

from agentlab2.metrics.models import SpanRecord

TRACES_JSONL = "traces.jsonl"


class JsonlSpanWriter:
    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self._path = os.path.join(run_dir, TRACES_JSONL)
        self._lock = threading.Lock()

    def write_span(self, span: ReadableSpan) -> None:
        context = span.get_span_context()  # type: ignore[no-untyped-call]
        parent = span.parent
        record = SpanRecord(
            trace_id=context.trace_id,
            span_id=context.span_id,
            parent_span_id=parent.span_id if parent else None,
            name=span.name,
            attributes=dict(span.attributes or {}),
            start_time=span.start_time,
            end_time=span.end_time,
            status=span.status.status_code.name,
        )
        self.write(record)

    def write(self, record: SpanRecord) -> None:
        with self._lock:
            with open(self._path, "a") as f:
                f.write(record.model_dump_json() + "\n")

    def scan_all(self) -> list[SpanRecord]:
        with self._lock:
            if not os.path.exists(self._path):
                return []
            with open(self._path) as f:
                return [SpanRecord.model_validate_json(line) for line in f if line.strip()]

    @staticmethod
    def flush() -> bool:
        # For now we just open the file on demand
        return True

    def close(self) -> None:
        # For now we just open the file on demand
        pass
