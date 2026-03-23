from __future__ import annotations

from chonkie import OverlapRefinery, RecursiveChunker

from .normalize import normalize_text


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")


def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 200) -> list[tuple[int, int, str]]:
    _validate_chunk_params(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunker = RecursiveChunker(tokenizer="character", chunk_size=chunk_size)
    chunks = chunker.chunk(normalized)

    if chunk_overlap > 0:
        overlap_refinery = OverlapRefinery(
            tokenizer="character",
            context_size=chunk_overlap,
            mode="token",
            method="suffix",
            merge=True,
            inplace=False,
        )
        chunks = overlap_refinery.refine(chunks)

    return [(chunk.start_index, chunk.end_index, chunk.text) for chunk in chunks]
