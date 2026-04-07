from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException

from stage7_1_retriever import Retriever, build_retriever_from_env
from stage7_1_retriever.models import RetrievedChunk
from stage7_2_reranker import CrossEncoderReranker, CrossEncoderRerankerConfig

from .config import AppConfig, load_config
from .schemas import HealthResponse, RetrieveRequest, RetrieveResponse

logger = logging.getLogger("stage7_3_retriever_api")


class RetrieverApiState:
    """Состояние приложения API."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever: Retriever | None = None
        self.reranker: CrossEncoderReranker | None = None

    async def startup(self) -> None:
        """Инициализирует retriever."""
        try:
            self.retriever = build_retriever_from_env()
            logger.info("Retriever initialized successfully")
        except Exception as exc:  # noqa: BLE001
            self.retriever = None
            logger.exception("Retriever initialization failed: %s", exc)

        self.reranker = CrossEncoderReranker(
            config=CrossEncoderRerankerConfig(
                enabled=self.config.reranker.enabled,
                model_name=self.config.reranker.model_name,
                top_n=self.config.reranker.top_n,
                batch_size=self.config.reranker.batch_size,
                timeout_sec=self.config.reranker.timeout_sec,
                max_length=self.config.reranker.max_length,
                device=self.config.reranker.device,
            )
        )
        if self.config.reranker.enabled:
            try:
                await self.reranker.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reranker initialization failed, disabled for runtime: %s", exc)
                self.reranker = None

    async def shutdown(self) -> None:
        """Закрывает ресурсы retriever."""
        if self.retriever is not None:
            await self.retriever.close()
            logger.info("Retriever resources closed")


config = load_config()
state = RetrieverApiState(config=config)
logging.basicConfig(
    level=config.api.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await state.startup()
    try:
        yield
    finally:
        await state.shutdown()


app = FastAPI(title="Retriever API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Проверяет готовность API."""
    return HealthResponse(
        status="ok" if state.retriever is not None else "degraded",
        retriever_ready=state.retriever is not None,
        reranker_enabled=state.config.reranker.enabled,
        reranker_ready=state.reranker is not None,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    """Возвращает найденные чанки по запросу."""
    if state.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not ready")

    start = time.perf_counter()
    final_top_k = payload.top_k
    retrieve_top_k = payload.top_k
    if state.config.reranker.enabled:
        if retrieve_top_k is None:
            retrieve_top_k = state.config.reranker.top_n
        else:
            retrieve_top_k = max(retrieve_top_k, state.config.reranker.top_n)

    try:
        chunks = await asyncio.wait_for(
            state.retriever.retrieve(
                query_text=payload.query,
                filters=payload.filters,
                top_k=retrieve_top_k,
            ),
            timeout=state.config.api.retrieve_timeout_sec,
        )
    except TimeoutError as exc:
        logger.warning("Retrieve timeout query_len=%s", len(payload.query))
        raise HTTPException(status_code=504, detail="Retrieve timeout") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieve failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieve failed") from exc

    rerank_start = time.perf_counter()
    if state.reranker is not None:
        chunks = await state.reranker.rerank(
            query_text=payload.query,
            chunks=chunks,
            final_top_k=final_top_k,
        )
    rerank_ms = int((time.perf_counter() - rerank_start) * 1000)

    latency_ms = int((time.perf_counter() - start) * 1000)
    limited = _clip_context(chunks, state.config.api.max_context_chars)
    logger.info(
        "Retrieve completed latency_ms=%s rerank_ms=%s query_len=%s results=%s",
        latency_ms,
        rerank_ms,
        len(payload.query),
        len(limited),
    )
    return RetrieveResponse(
        query=payload.query,
        top_k=payload.top_k or len(limited),
        latency_ms=latency_ms,
        results=[_chunk_to_dict(item) for item in limited],
    )


def _clip_context(chunks: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    clipped: list[RetrievedChunk] = []
    for chunk in chunks:
        text = chunk.text
        if len(text) > max_chars:
            text = text[:max_chars]
        clipped.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                text=text,
                source_scores=dict(chunk.source_scores),
                metadata=dict(chunk.metadata),
            )
        )
    return clipped


def _chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return asdict(chunk)

