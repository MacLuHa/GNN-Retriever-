from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Запрос на retrieval."""

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunkResponse(BaseModel):
    """Результат retrieval по чанку."""

    chunk_id: str
    score: float
    text: str
    source_scores: dict[str, float]
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    """Ответ retrieval endpoint."""

    query: str
    top_k: int
    latency_ms: int
    results: list[RetrievedChunkResponse]


class HealthResponse(BaseModel):
    """Ответ endpoint здоровья."""

    status: str
    retriever_ready: bool
    reranker_enabled: bool
    reranker_ready: bool

