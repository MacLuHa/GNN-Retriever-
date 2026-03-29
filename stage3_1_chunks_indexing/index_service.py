from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import GroupCoordinatorNotAvailableError, NodeNotReadyError

from .config import AppConfig
from .document_builder import build_elasticsearch_body
from .es_indexer import ElasticsearchIndexer
from .models import ChunkMessage, VectorizeMessageValue

logger = logging.getLogger("stage3_1_chunks_indexing")

_KAFKA_START_MAX_ATTEMPTS = 60
_KAFKA_START_RETRY_SEC = 2.0


class ChunkIndexService:
    """Читает чанки из Kafka, пишет в Elasticsearch, шлёт задачу векторизации."""

    def __init__(
        self,
        config: AppConfig,
        indexer: ElasticsearchIndexer,
        producer: AIOKafkaProducer,
    ) -> None:
        """Принимает конфигурацию, индексатор ES и producer для documents.vectorize."""
        self._config = config
        self._indexer = indexer
        self._producer = producer

    async def run_forever(self) -> None:
        """Создаёт индекс при необходимости, затем бесконечно читает сообщения до отмены."""
        await self._indexer.ensure_index()
        consumer = await self._start_consumer_with_retry()
        logger.info(
            "Listening topic=%s group=%s",
            self._config.kafka.chunks_topic,
            self._config.kafka.consumer_group,
        )
        try:
            async for msg in consumer:
                await self._process_record(msg.value)
        finally:
            await consumer.stop()

    async def _start_consumer_with_retry(self) -> AIOKafkaConsumer:
        """Ждёт готовности брокера/координатора (KRaft и single-broker стартуют с задержкой)."""
        for attempt in range(1, _KAFKA_START_MAX_ATTEMPTS + 1):
            consumer = AIOKafkaConsumer(
                self._config.kafka.chunks_topic,
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
        """Валидирует чанк, индексирует в ES, публикует в топик векторизации."""
        if not isinstance(payload, dict):
            logger.error("Chunk message must be JSON object, got %s", type(payload).__name__)
            return
        try:
            chunk = ChunkMessage.model_validate(payload)
        except Exception:
            logger.exception("Invalid chunk message payload")
            return
        body = build_elasticsearch_body(chunk)
        # Контракт: _id в ES = chunk_id → es_doc_id в Kafka всегда равен chunk_id (идемпотентность).
        es_doc_id = await self._indexer.index_document(chunk.chunk_id, body)
        await self._publish_vectorize(chunk, es_doc_id)

    async def _publish_vectorize(self, chunk: ChunkMessage, es_doc_id: str) -> None:
        """Kafka value — идентификаторы и текст чанка; Kafka record headers — trace_id."""
        trace_id = str(uuid.uuid4())
        value = VectorizeMessageValue(
            doc_id=str(chunk.doc_id),
            chunk_id=chunk.chunk_id,
            version_id=chunk.version_id,
            es_doc_id=es_doc_id,
            text=chunk.text,
        ).model_dump(mode="json")
        kafka_headers = [("trace_id", trace_id.encode("utf-8"))]
        await self._producer.send_and_wait(
            self._config.kafka.vectorize_topic,
            value,
            headers=kafka_headers,
        )
