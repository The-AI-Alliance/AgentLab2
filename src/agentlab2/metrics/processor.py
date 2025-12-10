import zipfile
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan

from agentlab2.metrics.models import SpanRecord
from agentlab2.metrics.store import JsonlSpanWriter

AL2_TYPE = "al2.type"
AL2_NAME = "al2.name"
AL2_EXPERIMENT = "al2.experiment"

TYPE_EXPERIMENT = "experiment"
TYPE_EPISODE = "episode"
TYPE_STEP = "step"


def _is_descendant_of(span: SpanRecord, ancestor_id: int, span_by_id: dict[int, SpanRecord]) -> bool:
    current = span
    while current.parent_span_id is not None:
        if current.parent_span_id == ancestor_id:
            return True
        parent = span_by_id.get(current.parent_span_id)
        if parent is None:
            break
        current = parent
    return False


def _zip_dir(dir_path: Path) -> Path:
    zip_path = dir_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in dir_path.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(dir_path.parent))
    return zip_path


class TraceProcessor:
    """Exports episode spans to experiment/episode/step_{N}.json hierarchy."""

    def __init__(self, run_dir: str) -> None:
        self._run_dir = Path(run_dir)
        self._store = JsonlSpanWriter(run_dir)

    def export_episode(self, episode_span: ReadableSpan) -> Path:
        """Export and zip a single episode when it completes."""
        attrs = dict(episode_span.attributes or {})
        episode_name = str(attrs.get(AL2_NAME, episode_span.name))
        experiment_name = str(attrs.get(AL2_EXPERIMENT, "default"))
        episode_span_id = episode_span.get_span_context().span_id  # type: ignore[no-untyped-call]

        all_spans = self._store.scan_all()
        span_by_id = {s.span_id: s for s in all_spans}

        steps = [
            s
            for s in all_spans
            if s.attributes.get(AL2_TYPE) == TYPE_STEP and _is_descendant_of(s, episode_span_id, span_by_id)
        ]
        steps.sort(key=lambda s: s.start_time or 0)

        episode_dir = self._run_dir / experiment_name / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        for i, step in enumerate(steps):
            (episode_dir / f"step_{i}.json").write_text(step.model_dump_json(indent=2))

        return _zip_dir(episode_dir)
