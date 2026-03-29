from __future__ import annotations

import asyncio
import logging
import math

import httpx

from .config import OllamaConfig

logger = logging.getLogger("stage3_2_chunks_vectorizing")


class OllamaEmbedder:
    """Генерирует embedding через Ollama."""

    def __init__(self, config: OllamaConfig) -> None:
        """Инициализирует клиент генерации embedding."""
        self._config = config

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Вычисляет косинусную близость между двумя векторами."""
        if not left or not right:
            raise ValueError("Vectors for cosine similarity must not be empty")

        size = min(len(left), len(right))
        left_slice = left[:size]
        right_slice = right[:size]

        dot = sum(x * y for x, y in zip(left_slice, right_slice))
        left_norm = math.sqrt(sum(x * x for x in left_slice))
        right_norm = math.sqrt(sum(y * y for y in right_slice))
        if left_norm == 0.0 or right_norm == 0.0:
            raise ValueError("Vectors for cosine similarity must be non-zero")
        return dot / (left_norm * right_norm)

    async def embed(self, text: str) -> list[float]:
        """Возвращает embedding для входного текста."""
        if not text.strip():
            raise ValueError("Text for embedding must not be empty")

        last_error: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                timeout = httpx.Timeout(self._config.timeout_sec)
                async with httpx.AsyncClient(base_url=self._config.base_url, timeout=timeout) as client:
                    truncated_response = await client.post(
                        "/api/embed",
                        json={"model": self._config.model_name, "input": text, "truncate": True, "dimensions": self._config.embedding_dim},
                    )
                    full_response = await client.post(
                        "/api/embed",
                        json={"model": self._config.model_name, "input": text, "truncate": False},
                    )

                    truncated_response.raise_for_status()
                    full_response.raise_for_status()

                    truncated_payload = truncated_response.json()
                    full_payload = full_response.json()
                    truncated_embeddings = truncated_payload.get("embeddings")
                    full_embeddings = full_payload.get("embeddings")
                    if (
                        not truncated_embeddings
                        or not isinstance(truncated_embeddings, list)
                        or len(truncated_embeddings) == 0
                    ):
                        raise ValueError("Ollama returned empty embeddings")
                    if not full_embeddings or not isinstance(full_embeddings, list) or len(full_embeddings) == 0:
                        raise ValueError("Ollama returned empty full embeddings")

                    truncated_embedding = truncated_embeddings[0]
                    full_embedding = full_embeddings[0]

                    similarity = self._cosine_similarity(truncated_embedding, full_embedding)
                    logger.info(
                        "Embedding similarity truncated_vs_full=%.6f dims=%s full_dims=%s",
                        similarity,
                        len(truncated_embedding),
                        len(full_embedding),
                    )
                    return truncated_embedding
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Embedding request failed attempt=%s/%s: %s",
                    attempt,
                    self._config.max_attempts,
                    exc,
                )
                if attempt < self._config.max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))

        raise RuntimeError("Failed to get embedding from Ollama") from last_error
