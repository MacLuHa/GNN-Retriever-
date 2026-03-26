from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from .config import QdrantConfig
from .models import VectorizeMessage

logger = logging.getLogger("stage3_2_chunks_vectorizing")


_DISTANCE_MAP: dict[str, models.Distance] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}


class QdrantStore:
    """Сохраняет embedding и metadata в Qdrant."""

    def __init__(self, client: AsyncQdrantClient, config: QdrantConfig) -> None:
        """Инициализирует хранилище векторов."""
        self._client = client
        self._config = config
        self._collection_ready = False
        self._vector_size: int | None = None

    async def upsert_vector(self, message: VectorizeMessage, vector: list[float]) -> str:
        """Делает idempotent upsert вектора и payload."""
        if not vector:
            raise ValueError("Vector must not be empty")

        await self._ensure_collection(len(vector))
        vector_id = self._build_vector_id(message)

        payload: dict[str, Any] = {
            "doc_id": message.doc_id,
            "chunk_id": message.chunk_id,
            "version_id": message.version_id,
            "es_doc_id": message.es_doc_id,
        }
        point = models.PointStruct(id=vector_id, vector=vector, payload=payload)
        await self._run_with_retry(
            self._client.upsert,
            collection_name=self._config.collection_name,
            points=[point],
            wait=True,
        )
        return vector_id

    async def _ensure_collection(self, vector_size: int) -> None:
        """Создаёт коллекцию при необходимости и проверяет размерность."""
        if self._collection_ready:
            if self._vector_size != vector_size:
                raise ValueError(
                    f"Vector size mismatch, expected {self._vector_size}, got {vector_size}"
                )
            return

        if await self._client.collection_exists(self._config.collection_name):
            info = await self._client.get_collection(self._config.collection_name)
            params = info.config.params.vectors
            existing_size = params.size if isinstance(params, models.VectorParams) else None
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Collection vector size mismatch, expected {existing_size}, got {vector_size}"
                )
            self._vector_size = existing_size or vector_size
            self._collection_ready = True
            return

        distance = _DISTANCE_MAP.get(self._config.distance.lower())
        if distance is None:
            raise ValueError(f"Unsupported Qdrant distance: {self._config.distance}")

        await self._run_with_retry(
            self._client.create_collection,
            collection_name=self._config.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        self._vector_size = vector_size
        self._collection_ready = True
        logger.info(
            "Qdrant collection created name=%s size=%s distance=%s",
            self._config.collection_name,
            vector_size,
            self._config.distance,
        )

    async def _run_with_retry(self, func, **kwargs):  # type: ignore[no-untyped-def]
        """Выполняет операцию Qdrant с ограниченным retry."""
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await func(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Qdrant operation failed attempt=%s/%s: %s",
                    attempt,
                    self._config.max_attempts,
                    exc,
                )
                if attempt < self._config.max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))
        raise RuntimeError("Qdrant operation failed after retries") from last_error

    @staticmethod
    def _build_vector_id(message: VectorizeMessage) -> str:
        """Строит стабильный идентификатор вектора."""
        seed = f"{message.chunk_id}:{message.version_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
