from __future__ import annotations

import asyncio
import logging
import math
from typing import Protocol

import httpx


class OllamaEmbeddingConfigLike(Protocol):
    """Минимальный интерфейс конфига для генерации embedding."""

    base_url: str
    model_name: str
    timeout_sec: float
    max_attempts: int
    embedding_dim: int


class OllamaEmbedder:
    """Генерирует embedding через Ollama."""

    def __init__(
        self,
        config: OllamaEmbeddingConfigLike,
        *,
        logger_name: str = "ollama_embedder",
        log_prefix: str = "Embedding",
        error_prefix: str = "embedding",
    ) -> None:
        self._config = config
        self._logger = logging.getLogger(logger_name)
        self._log_prefix = log_prefix
        self._error_prefix = error_prefix

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
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
                        json={
                            "model": self._config.model_name,
                            "input": text,
                            "truncate": True,
                            "dimensions": self._config.embedding_dim,
                        },
                    )
                    full_response = await client.post(
                        "/api/embed",
                        json={
                            "model": self._config.model_name,
                            "input": text,
                            "truncate": False,
                        },
                    )

                    truncated_response.raise_for_status()

                    truncated_payload = truncated_response.json()
                    
                    truncated_embeddings = truncated_payload.get("embeddings")
                    
                    truncated_embedding = truncated_embeddings[0]
                   
                    return truncated_embedding
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.warning(
                    "%s request failed attempt=%s/%s: %s",
                    self._log_prefix,
                    attempt,
                    self._config.max_attempts,
                    exc,
                )
                if attempt < self._config.max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))

        raise RuntimeError(f"Failed to get {self._error_prefix} from Ollama") from last_error
