from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any

from stage2_ingestion.normalize import normalize_text

from .models import ChunkMessage


def _parse_effective_date(raw: Any) -> str | None:
    """Приводит effective_date из метаданных к строке даты для Elasticsearch."""
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw.isoformat()
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        return s[:10] if len(s) >= 10 else None
    return None


def _coerce_bool(raw: Any, default: bool = True) -> bool:
    """Преобразует значение метаданных к bool."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in ("1", "true", "yes", "y")
    return bool(raw)


def _entity_ids_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Извлекает список id сущностей (Neo4j) из metadata.entity_ids."""
    raw = metadata.get("entity_ids")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    return []


def build_elasticsearch_body(msg: ChunkMessage) -> dict[str, Any]:
    """Формирует тело документа для индекса Elasticsearch из сообщения-чанка."""
    meta = msg.metadata
    normalized = normalize_text(msg.text)
    exact_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return {
        "chunk_id": msg.chunk_id,
        "doc_id": str(msg.doc_id),
        "version_id": msg.version_id,
        "section_id": str(meta.get("section_id") or ""),
        "text": msg.text,
        "normalized_text": normalized,
        "language": str(meta.get("language") or "und"),
        "jurisdiction": str(meta.get("jurisdiction") or ""),
        "source_type": str(meta.get("source_type") or "web"),
        "is_canonical": _coerce_bool(meta.get("is_canonical"), default=True),
        "page": msg.page,
        "span_start": msg.span_start,
        "span_end": msg.span_end,
        "effective_date": _parse_effective_date(meta.get("effective_date")),
        "exact_hash": exact_hash,
        "entity_ids": _entity_ids_from_metadata(meta),
    }
