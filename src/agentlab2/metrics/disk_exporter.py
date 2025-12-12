import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agentlab2.metrics.processor import AL2_TYPE, TYPE_EPISODE, TraceProcessor
from agentlab2.metrics.store import JsonlSpanWriter


logger = logging.getLogger(__name__)

MAX_PENDING_EXPORTS = 10


class DiskSpanExporter(SpanExporter):
    def __init__(self, run_dir: str) -> None:
        self._store = JsonlSpanWriter(run_dir)
        self._processor = TraceProcessor(run_dir)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: list[Future[Path]] = []


    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self._store.write_span(span)

        for span in spans:
            if dict(span.attributes or {}).get(AL2_TYPE) == TYPE_EPISODE:
                future = self._executor.submit(self._processor.export_episode, span)
                self._pending.append(future)

        self._pending = [f for f in self._pending if not f.done()]

        if len(self._pending) >= MAX_PENDING_EXPORTS:
            logger.warning(
                "Episode export backlog reached %d, forcing flush to prevent unbounded growth",
                len(self._pending),
            )
            self.force_flush()

        return SpanExportResult.SUCCESS


    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._store.flush()
        for future in self._pending:
            future.result(timeout=timeout_millis / 1000)
        self._pending.clear()
        return True


    def shutdown(self) -> None:
        self.force_flush()
        self._executor.shutdown(wait=True)
        self._store.close()
