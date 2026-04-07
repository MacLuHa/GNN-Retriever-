from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from stage7_1_retriever.models import RetrievedChunk

logger = logging.getLogger("stage7_2_reranker")


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    """Параметры cross-encoder реранкера."""

    enabled: bool
    model_name: str
    top_n: int
    batch_size: int
    timeout_sec: float
    max_length: int
    device: str


class CrossEncoderReranker:
    """Реранкер кандидатов через cross-encoder."""

    def __init__(self, config: CrossEncoderRerankerConfig) -> None:
        self._config = config
        self._model = None

    async def load(self) -> None:
        """Загружает модель реранкера в память."""
        if not self._config.enabled:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc

        kwargs: dict[str, object] = {
            "model_name": self._config.model_name,
            "max_length": self._config.max_length,
        }
        if self._config.device and self._config.device.lower() != "auto":
            kwargs["device"] = self._config.device

        self._model = CrossEncoder(**kwargs)
        logger.info(
            "Cross-encoder reranker loaded model=%s device=%s",
            self._config.model_name,
            self._config.device,
        )

    async def rerank(
        self,
        query_text: str,
        chunks: list[RetrievedChunk],
        final_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Пересортировывает кандидатов по cross-encoder score."""
        if not self._config.enabled or self._model is None or not chunks:
            return chunks

        candidate_count = min(len(chunks), max(1, self._config.top_n))
        head = chunks[:candidate_count]
        tail = chunks[candidate_count:]

        pairs: list[tuple[str, str]] = []
        for chunk in head:
            text = chunk.text.strip()
            if not text:
                text = str(chunk.metadata.get("text") or "")
            pairs.append((query_text, text))

        try:
            raw_scores = await asyncio.wait_for(
                asyncio.to_thread(self._predict_scores, pairs),
                timeout=self._config.timeout_sec,
            )
        except TimeoutError:
            logger.warning("Cross-encoder rerank timeout, fallback to retriever ranking")
            return chunks
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder rerank failed, fallback to retriever ranking: %s", exc)
            return chunks

        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(head, raw_scores, strict=False):
            updated = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=float(score),
                text=chunk.text,
                source_scores={**chunk.source_scores, "cross_encoder": float(score)},
                metadata={**chunk.metadata, "rerank_score": float(score)},
            )
            reranked.append(updated)

        reranked.sort(key=lambda item: item.score, reverse=True)
        merged = reranked + tail
        if final_top_k is not None:
            return merged[: max(1, final_top_k)]
        return merged

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            return []

        output = self._model.predict(
            pairs,
            batch_size=max(1, self._config.batch_size),
            show_progress_bar=False,
        )
        if hasattr(output, "tolist"):
            return [float(item) for item in output.tolist()]
        return [float(item) for item in output]

