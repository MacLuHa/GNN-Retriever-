from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from stage7_1_retriever import Retriever, build_retriever_from_env
from stage7_1_retriever.models import RetrievedChunk
from stage7_2_reranker import CrossEncoderReranker, CrossEncoderRerankerConfig

from .config import AppConfig, load_config
from .schemas import HealthResponse, RetrieveRequest, RetrieveResponse

logger = logging.getLogger("stage7_3_retriever_api")


class _LangfuseTracer:
    """Безопасная обертка для отправки traces/spans в Langfuse."""

    def __init__(self, *, enabled: bool, host: str, public_key: str, secret_key: str) -> None:
        self._client: Any | None = None
        if not enabled:
            return
        if not public_key or not secret_key:
            logger.warning("Langfuse enabled but keys are empty, tracing disabled")
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                host=host,
                public_key=public_key,
                secret_key=secret_key,
            )
            logger.info("Langfuse tracing enabled host=%s", host)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse initialization failed: %s", exc)
            self._client = None

    def trace(
        self,
        *,
        trace_id: str,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        if self._client is None:
            return None
        try:
            return self._client.trace(
                id=trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse trace error: %s", exc)
            return None

    def span(
        self,
        *,
        trace: Any | None,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        if trace is None:
            return None
        try:
            return trace.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse span error: %s", exc)
            return None

    def end_span(
        self,
        span: Any | None,
        *,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        if span is None:
            return
        try:
            kwargs: dict[str, Any] = {}
            if output_data is not None:
                kwargs["output"] = output_data
            if metadata:
                kwargs["metadata"] = metadata
            if level is not None:
                kwargs["level"] = level
            span.end(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse span end error: %s", exc)

    def update_trace(
        self,
        trace: Any | None,
        *,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if trace is None:
            return
        try:
            kwargs: dict[str, Any] = {}
            if output_data is not None:
                kwargs["output"] = output_data
            if metadata:
                kwargs["metadata"] = metadata
            trace.update(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse trace update error: %s", exc)

    def flush(self) -> None:
        if self._client is None:
            return
        flush_method = getattr(self._client, "flush", None)
        if callable(flush_method):
            try:
                flush_method()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse flush error: %s", exc)


class RetrieverApiState:
    """Состояние приложения API."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever: Retriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.langfuse = _LangfuseTracer(
            enabled=self.config.langfuse.enabled,
            host=self.config.langfuse.host,
            public_key=self.config.langfuse.public_key,
            secret_key=self.config.langfuse.secret_key,
        )

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
        self.langfuse.flush()


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
async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    """Возвращает найденные чанки по запросу."""
    trace_id = request.headers.get("X-Trace-Id") or str(uuid4())
    chat_session_id = request.headers.get("X-Chat-Session-Id") or "unknown"
    trace = state.langfuse.trace(
        trace_id=trace_id,
        name="retriever_api.retrieve",
        input_data={
            "query": payload.query,
            "filters": payload.filters,
            "top_k": payload.top_k,
        },
        metadata={
            "service": "stage7_3_retriever_api",
            "chat_session_id": chat_session_id,
        },
    )

    if state.retriever is None:
        state.langfuse.update_trace(
            trace,
            metadata={
                "error": "retriever_not_ready",
                "chat_session_id": chat_session_id,
            },
        )
        raise HTTPException(status_code=503, detail="Retriever is not ready")

    start = time.perf_counter()
    final_top_k = payload.top_k
    retrieve_top_k = payload.top_k
    if state.config.reranker.enabled:
        if retrieve_top_k is None:
            retrieve_top_k = state.config.reranker.top_n
        else:
            retrieve_top_k = max(retrieve_top_k, state.config.reranker.top_n)

    retrieval_span = state.langfuse.span(
        trace=trace,
        name="knowledge_base_retrieval",
        input_data={"query": payload.query, "top_k": retrieve_top_k, "filters": payload.filters},
        metadata={"stage": "retrieval"},
    )
    try:
        chunks = await asyncio.wait_for(
            state.retriever.retrieve(
                query_text=payload.query,
                filters=payload.filters,
                top_k=retrieve_top_k,
            ),
            timeout=state.config.api.retrieve_timeout_sec,
        )
        state.langfuse.end_span(
            retrieval_span,
            output_data={"results_count": len(chunks)},
            metadata={"stage": "retrieval"},
        )
    except TimeoutError as exc:
        logger.warning("Retrieve timeout query_len=%s", len(payload.query))
        state.langfuse.end_span(
            retrieval_span,
            metadata={"error": "retrieve_timeout", "stage": "retrieval"},
            level="ERROR",
        )
        state.langfuse.update_trace(trace, metadata={"error": "retrieve_timeout"})
        raise HTTPException(status_code=504, detail="Retrieve timeout") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieve failed: %s", exc)
        state.langfuse.end_span(
            retrieval_span,
            metadata={"error": str(exc), "stage": "retrieval"},
            level="ERROR",
        )
        state.langfuse.update_trace(trace, metadata={"error": "retrieve_failed"})
        raise HTTPException(status_code=500, detail="Retrieve failed") from exc

    rerank_start = time.perf_counter()
    rerank_span = state.langfuse.span(
        trace=trace,
        name="rerank",
        input_data={"initial_results_count": len(chunks), "enabled": state.reranker is not None},
        metadata={"stage": "rerank"},
    )
    if state.reranker is not None:
        chunks = await state.reranker.rerank(
            query_text=payload.query,
            chunks=chunks,
            final_top_k=final_top_k,
        )
    rerank_ms = int((time.perf_counter() - rerank_start) * 1000)
    state.langfuse.end_span(
        rerank_span,
        output_data={"results_count": len(chunks), "rerank_ms": rerank_ms},
        metadata={"stage": "rerank"},
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    clip_span = state.langfuse.span(
        trace=trace,
        name="clip_context",
        input_data={"max_context_chars": state.config.api.max_context_chars, "results_count": len(chunks)},
        metadata={"stage": "response"},
    )
    limited = _clip_context(chunks, state.config.api.max_context_chars)
    state.langfuse.end_span(
        clip_span,
        output_data={"results_count": len(limited)},
        metadata={"stage": "response"},
    )
    logger.info(
        "Retrieve completed latency_ms=%s rerank_ms=%s query_len=%s results=%s",
        latency_ms,
        rerank_ms,
        len(payload.query),
        len(limited),
    )
    response = RetrieveResponse(
        query=payload.query,
        top_k=payload.top_k or len(limited),
        latency_ms=latency_ms,
        results=[_chunk_to_dict(item) for item in limited],
    )
    state.langfuse.update_trace(
        trace,
        output_data={
            "latency_ms": latency_ms,
            "rerank_ms": rerank_ms,
            "results_count": len(limited),
        },
        metadata={
            "service": "stage7_3_retriever_api",
            "chat_session_id": chat_session_id,
            "status": "ok",
        },
    )
    return response


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

