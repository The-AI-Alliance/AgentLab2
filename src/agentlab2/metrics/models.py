from typing import Any

from pydantic import BaseModel


class SpanRecord(BaseModel):
    trace_id: int
    span_id: int
    parent_span_id: int | None
    name: str
    attributes: dict[str, Any]
    start_time: int | None
    end_time: int | None
    status: str | int
