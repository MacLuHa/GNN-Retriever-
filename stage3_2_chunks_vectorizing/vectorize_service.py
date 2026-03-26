from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import GroupCoordinatorNotAvailableError, NodeNotReadyError

from .config import AppConfig
from .models import GraphMessage, VectorizeMessage
from .ollama_embedder import OllamaEmbedder
from .qdrant_store import QdrantStore

logger = logging.getLogger("stage3_2_chunks_vectorizing")

_KAFKA_START_MAX_ATTEMPTS = 60
_KAFKA_START_RETRY_SEC = 2.0


class ChunkVectorizeService:
    """Читает задачи векторизации и пишет результат в Qdrant."""

    def __init__(
        self,
        config: AppConfig,
        embedder: OllamaEmbedder,
        qdrant_store: QdrantStore,
        producer: AIOKafkaProducer,
    ) -> None:
        """Инициализирует зависимости сервиса."""
        self._config = config
        self._embedder = embedder
        self._qdrant_store = qdrant_store
        self._producer = producer

    async def run_forever(self) -> None:
        """Запускает бесконечный цикл обработки сообщений Kafka."""
        consumer = await self._start_consumer_with_retry()
        logger.info(
            "Listening topic=%s group=%s",
            self._config.kafka.vectorize_topic,
            self._config.kafka.consumer_group,
        )
        try:
            async for msg in consumer:
                await self._process_record(msg.value)
        finally:
            await consumer.stop()

    async def _start_consumer_with_retry(self) -> AIOKafkaConsumer:
        """Запускает consumer с retry при позднем старте Kafka."""
        for attempt in range(1, _KAFKA_START_MAX_ATTEMPTS + 1):
            consumer = AIOKafkaConsumer(
                self._config.kafka.vectorize_topic,
                bootstrap_servers=self._config.kafka.bootstrap_servers,
                group_id=self._config.kafka.consumer_group,
                enable_auto_commit=True,
                auto_offset_reset=self._config.kafka.auto_offset_reset,
                value_deserializer=self._deserialize_value,
            )
            try:
                await consumer.start()
                return consumer
            except (GroupCoordinatorNotAvailableError, NodeNotReadyError) as exc:
                logger.warning(
                    "Kafka not ready for consumer group (attempt %s/%s): %s",
                    attempt,
                    _KAFKA_START_MAX_ATTEMPTS,
                    exc,
                )
                try:
                    await consumer.stop()
                except Exception:
                    pass
                if attempt == _KAFKA_START_MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(_KAFKA_START_RETRY_SEC)
            except BaseException:
                try:
                    await consumer.stop()
                except Exception:
                    pass
                raise
        raise RuntimeError("Kafka consumer start failed after retries")

    @staticmethod
    def _deserialize_value(raw: bytes | None) -> Any:
        """Десериализует JSON значение сообщения Kafka."""
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    async def _process_record(self, payload: Any) -> None:
        """Обрабатывает одно сообщение векторизации."""
        if not isinstance(payload, dict):
            logger.error("Vectorize message must be JSON object, got %s", type(payload).__name__)
            return

        try:
            message = VectorizeMessage.model_validate(payload)
        except Exception:
            logger.exception("Invalid vectorize message payload")
            return

        vector = await self._embedder.embed(message.text)
        vector_id = await self._qdrant_store.upsert_vector(message, vector)
        await self._publish_graph_event(message, vector_id)
        logger.info("Chunk vectorized chunk_id=%s vector_id=%s", message.chunk_id, vector_id)

    async def _publish_graph_event(self, message: VectorizeMessage, vector_id: str) -> None:
        """Публикует событие следующего шага в topic documents.graph."""
        trace_id = str(uuid.uuid4())
        value = GraphMessage(
            doc_id=message.doc_id,
            chunk_id=message.chunk_id,
            version_id=message.version_id,
            es_doc_id=message.es_doc_id,
            vectorId=vector_id,
        ).model_dump(mode="json")
        kafka_headers = [("trace_id", trace_id.encode("utf-8"))]
        await self._producer.send_and_wait(
            self._config.kafka.graph_topic,
            value,
            headers=kafka_headers,
        )
