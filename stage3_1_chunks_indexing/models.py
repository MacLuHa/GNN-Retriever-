from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkMessage(BaseModel):
    """Сообщение из топика documents.chunks."""

    chunk_id: str
    doc_id: str
    version_id: str
    title: str
    page: int | None = None
    span_start: int
    span_end: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorizeMessageValue(BaseModel):
    """Содержимое Kafka value (JSON) для топика векторизации."""

    doc_id: str
    chunk_id: str
    version_id: str
    es_doc_id: str
    text: str
