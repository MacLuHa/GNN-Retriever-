from __future__ import annotations

import base64
import uuid

from .chunking import chunk_text
from .models import Chunk, DocumentMessage
from .parsers import parse_document


def _build_chunk_id() -> str:
    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())

    from uuid6 import uuid7

    return str(uuid7())


def _extract_pages(doc: DocumentMessage) -> list[tuple[int, str]]:
    source = doc.source_type.lower()
    if source == "text":
        if doc.text is not None:
            return [(1, doc.text)]
        if doc.content_base64 is None:
            raise ValueError("For source_type='text', provide 'text' or 'content_base64'")
        text = base64.b64decode(doc.content_base64).decode("utf-8", errors="replace")
        return [(1, text)]

    if doc.content_base64 is None:
        raise ValueError(f"For source_type='{source}', 'content_base64' is required")

    raw_bytes = base64.b64decode(doc.content_base64)
    return parse_document(raw_bytes, source)


def process_document_message(
    payload: dict,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    doc = DocumentMessage.model_validate(payload)
    pages = _extract_pages(doc)

    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        for span_start, span_end, chunk_value in chunk_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunks.append(
                Chunk(
                    chunk_id=_build_chunk_id(),
                    doc_id=doc.doc_id,
                    version_id=doc.version_id,
                    title=doc.title,
                    page=page_num,
                    span_start=span_start,
                    span_end=span_end,
                    text=chunk_value,
                    metadata=doc.metadata,
                )
            )

    return chunks
