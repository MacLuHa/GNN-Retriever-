from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrieverQuery:
    """Запрос к ретриверу."""

    text: str
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None


@dataclass
class RetrievedChunk:
    """Результат поиска по чанку."""

    chunk_id: str
    score: float
    text: str = ""
    source_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
