from __future__ import annotations

import asyncio
import logging

import httpx

from .config import OllamaConfig

logger = logging.getLogger("stage3_2_chunks_vectorizing")


class OllamaEmbedder:
    """Генерирует embedding через Ollama."""

    def __init__(self, config: OllamaConfig) -> None:
        """Инициализирует клиент генерации embedding."""
        self._config = config

    async def embed(self, text: str) -> list[float]:
        """Возвращает embedding для входного текста."""
        if not text.strip():
            raise ValueError("Text for embedding must not be empty")

        last_error: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                timeout = httpx.Timeout(self._config.timeout_sec)
                async with httpx.AsyncClient(base_url=self._config.base_url, timeout=timeout) as client:
                    response = await client.post(
                        "/api/embed",
                        json={"model": self._config.model_name, "input": text, "truncate": True, "dimensions": self._config.embedding_dim},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    embeddings = payload.get("embeddings")
                    print(embeddings)
                    if not embeddings or not isinstance(embeddings, list) or len(embeddings) == 0:
                        raise ValueError("Ollama returned empty embeddings")
                    return embeddings[0]
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
