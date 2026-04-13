"""
title: Retriever Pipeline
author: local
version: 0.1.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

logger = logging.getLogger("openwebui.retriever_pipeline")

# SDK и сервер Langfuse отклоняют события при невалидном environment (см. langfuse.client.ENVIRONMENT_PATTERN).
_LANGFUSE_ENV_PATTERN = re.compile(r"^(?!langfuse)[a-z0-9-_]+$")


class _LangfuseTracer:
    """Безопасная обертка для Langfuse SDK с graceful fallback."""

    def __init__(self, *, enabled: bool, host: str, public_key: str, secret_key: str) -> None:
        self._client: Any | None = None
        if not enabled:
            return
        if not public_key or not secret_key:
            logger.warning("Langfuse enabled but keys are empty, tracing disabled")
            return
        host = host.rstrip("/")
        raw_trace_env = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
        lf_kwargs: dict[str, Any] = {}
        if raw_trace_env:
            if _LANGFUSE_ENV_PATTERN.match(raw_trace_env):
                lf_kwargs["environment"] = raw_trace_env
            else:
                logger.warning(
                    "LANGFUSE_TRACING_ENVIRONMENT=%r is invalid for Langfuse ingestion; using 'pipeline'. "
                    "Pattern: lowercase letters, digits, hyphen, underscore; must not start with 'langfuse'.",
                    raw_trace_env,
                )
                lf_kwargs["environment"] = "pipeline"
        debug = os.getenv("LANGFUSE_DEBUG", "").lower() in ("1", "true", "yes")
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                host=host,
                public_key=public_key,
                secret_key=secret_key,
                debug=debug,
                **lf_kwargs,
            )
            if not getattr(self._client, "enabled", True):
                logger.warning(
                    "Langfuse SDK is disabled (sample rate / keys); no events will be sent. "
                    "Check LANGFUSE_SAMPLE_RATE and keys in the pipelines container."
                )
            logger.info("Langfuse tracing enabled host=%s", host)
            if os.getenv("LANGFUSE_AUTH_CHECK", "true").lower() in ("1", "true", "yes"):
                try:
                    self._client.auth_check()
                    logger.info("Langfuse auth_check succeeded (keys match a project on this host)")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Langfuse auth_check failed — events will likely not appear in the UI: %s",
                        exc,
                        exc_info=True,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse initialization failed: %s", exc, exc_info=True)
            self._client = None

    def trace(
        self,
        *,
        trace_id: str,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Any | None:
        if self._client is None:
            return None
        try:
            return self._client.trace(
                id=trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
                session_id=session_id,
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse trace error: %s", exc, exc_info=True)
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
            return trace.span(name=name, input=input_data, metadata=metadata or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse span error: %s", exc, exc_info=True)
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
            logger.warning("Langfuse span end error: %s", exc, exc_info=True)

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
            logger.warning("Langfuse trace update error: %s", exc, exc_info=True)

    def flush(self) -> None:
        if self._client is None:
            return
        flush_method = getattr(self._client, "flush", None)
        if callable(flush_method):
            try:
                flush_method()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse flush failed: %s", exc)

    def has_client(self) -> bool:
        return self._client is not None


# Open WebUI может перезагружать модуль между inlet и outlet; у каждого нового Pipeline()
# был бы свой Langfuse() и свой dict — outlet вызывал flush() на пустом клиенте, а события
# оставались в буфере старого экземпляра.
_shared_langfuse: _LangfuseTracer | None = None
_pending_traces: dict[str, Any] = {}


def _get_shared_langfuse() -> _LangfuseTracer:
    global _shared_langfuse
    if _shared_langfuse is None:
        _shared_langfuse = _LangfuseTracer(
            enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        )
    return _shared_langfuse


def _correlation_keys_from_body(body: dict[str, Any]) -> list[str]:
    """Ключи inlet→outlet. Не используем top-level id — в outlet это id сообщения, не совпадает с запросом inlet."""
    keys: list[str] = []
    for key in ("chat_id", "conversation_id", "session_id"):
        val = body.get(key)
        if val is not None and str(val):
            keys.append(str(val))
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        for mk in ("chat_id", "conversation_id", "session_id", "_langfuse_trace_id"):
            val = metadata.get(mk)
            if val is not None and str(val):
                s = str(val)
                if s not in keys:
                    keys.append(s)
    return keys


def _register_pending_trace(keys: list[str], trace: Any) -> None:
    if trace is None:
        return
    for k in keys:
        if k:
            _pending_traces[k] = trace


def _take_pending_trace(body: dict[str, Any]) -> Any | None:
    keys = _correlation_keys_from_body(body)
    trace = None
    for k in keys:
        candidate = _pending_traces.get(k)
        if candidate is not None:
            trace = candidate
            break
    if trace is None:
        return None
    stale = [pk for pk, ref in list(_pending_traces.items()) if ref is trace]
    for pk in stale:
        _pending_traces.pop(pk, None)
    return trace


class Pipeline:
    """Pipeline Open WebUI с HTTP retrieval."""

    class Valves(BaseModel):
        """Связь filter → модели в Open WebUI (см. examples/filters в open-webui/pipelines)."""

        pipelines: list[str] = ["*"]
        priority: int = 10

    def __init__(self) -> None:
        self.type = "filter"
        self.name = "Retriever RAG"
        self.id = "retriever_pipeline"
        self.valves = self.Valves(pipelines=["*"], priority=10)
        self.retriever_api_url = os.getenv("RETRIEVER_API_URL", "http://stage7-3-retriever-api:8010")
        self.retriever_timeout_sec = float(os.getenv("RETRIEVER_TIMEOUT_SEC", "10"))
        self.retriever_top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
        self.context_max_chars = int(os.getenv("RETRIEVER_MAX_CONTEXT_CHARS", "1200"))
        self.context_max_chunks = int(os.getenv("RETRIEVER_MAX_CONTEXT_CHUNKS", "5"))
        self.langfuse = _get_shared_langfuse()

    async def inlet(self, body: dict[str, Any], user: dict[str, Any] | None = None) -> dict[str, Any]:
        print(f"inlet body: {body}")
        print(f"user: {user}")
        """Добавляет retrieval context в system message."""
        messages = body.get("messages", [])
        query = _extract_user_query(body)
        if not query:
            logger.warning("No user query found for retrieval")
            return body

        trace_id = _extract_trace_id(body) or str(uuid4())
        chat_session_id = _extract_chat_session_id(body, user)
        _store_trace_metadata(body, trace_id=trace_id, chat_session_id=chat_session_id)
        corr_keys = list(dict.fromkeys(_correlation_keys_from_body(body) + [trace_id]))
        history_metrics = _history_metrics(messages)
        trace = self.langfuse.trace(
            trace_id=trace_id,
            name="openwebui.turn",
            input_data={"query": query},
            metadata={
                "service": "openwebui-pipeline",
                "chat_session_id": chat_session_id,
                **history_metrics,
            },
            session_id=chat_session_id,
            user_id=(str(user.get("email") or user.get("id") or "") if isinstance(user, dict) else None)
            or None,
        )
        if trace is None and self.langfuse.has_client():
            logger.warning("Langfuse trace() returned None; events for this turn were not queued")
        _register_pending_trace(corr_keys, trace)
        if trace is not None and os.getenv("LANGFUSE_LOG_TRACE_URL", "").lower() in ("1", "true", "yes"):
            url_fn = getattr(trace, "get_trace_url", None)
            if callable(url_fn):
                try:
                    logger.info("Langfuse trace url=%s", url_fn())
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Langfuse get_trace_url failed: %s", exc)
        try:
            history_span = self.langfuse.span(
                trace=trace,
                name="history_parse",
                input_data=history_metrics,
                metadata={"stage": "history"},
            )
            self.langfuse.end_span(history_span, output_data=history_metrics, metadata={"stage": "history"})

            start = time.perf_counter()
            retrieve_span = self.langfuse.span(
                trace=trace,
                name="retriever_request",
                input_data={"query": query, "top_k": self.retriever_top_k},
                metadata={"stage": "retrieval"},
            )
            context = await asyncio.to_thread(self._fetch_context, query, trace_id, chat_session_id)
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.info("Retriever pipeline latency_ms=%s query_len=%s", latency_ms, len(query))
            self.langfuse.end_span(
                retrieve_span,
                output_data={"context_chars": len(context), "latency_ms": latency_ms},
                metadata={"stage": "retrieval"},
            )

            if not context:
                # Fallback: keep prompt untouched when retriever is unavailable or empty.
                llm_request_span = self.langfuse.span(
                    trace=trace,
                    name="llm_request",
                    input_data={"model": body.get("model"), "messages_count": len(messages)},
                    metadata={"stage": "llm_request", "fallback_without_context": True},
                )
                self.langfuse.end_span(llm_request_span, metadata={"stage": "llm_request"})
                return body

            inject_span = self.langfuse.span(
                trace=trace,
                name="context_injection",
                input_data={"context_chars": len(context)},
                metadata={"stage": "context"},
            )
            system_prefix = (
                "Use the retrieved context first. "
                "If context is insufficient, explicitly say so.\n\n"
                f"{context}"
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{system_prefix}\n\n{messages[0].get('content', '')}"
            else:
                messages.insert(0, {"role": "system", "content": system_prefix})
            body["messages"] = messages
            self.langfuse.end_span(
                inject_span,
                output_data={"messages_count": len(messages), "system_prefix_chars": len(system_prefix)},
                metadata={"stage": "context"},
            )
            llm_request_span = self.langfuse.span(
                trace=trace,
                name="llm_request",
                input_data={"model": body.get("model"), "messages_count": len(messages)},
                metadata={"stage": "llm_request", "fallback_without_context": False},
            )
            self.langfuse.end_span(llm_request_span, metadata={"stage": "llm_request"})
            return body
        finally:
            self.langfuse.flush()

    async def outlet(self, body: dict[str, Any], user: dict[str, Any] | None = None) -> dict[str, Any]:
        print(f"outlet body: {body}")
        print(f"user: {user}")
        """Фиксирует ответ модели как llm_response span."""
        trace = _take_pending_trace(body)
        if trace is None and self.langfuse.has_client():
            logger.warning(
                "Langfuse outlet: pending trace not found (tried keys %s); llm_response span skipped",
                _correlation_keys_from_body(body),
            )
        assistant_reply = _extract_assistant_response(body)
        response_span = self.langfuse.span(
            trace=trace,
            name="llm_response",
            input_data={"assistant_chars": len(assistant_reply)},
            metadata={"stage": "llm_response"},
        )
        self.langfuse.end_span(
            response_span,
            output_data={"assistant_preview": assistant_reply[:1000]},
            metadata={"stage": "llm_response"},
        )
        self.langfuse.update_trace(
            trace,
            output_data={"assistant_preview": assistant_reply[:1000]},
            metadata={"status": "ok"},
        )
        self.langfuse.flush()
        return body

    def _fetch_context(self, query: str, trace_id: str, chat_session_id: str) -> str:
        payload = {"query": query, "top_k": self.retriever_top_k, "filters": {}}
        try:
            request = urllib.request.Request(
                f"{self.retriever_api_url}/retrieve",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Trace-Id": trace_id,
                    "X-Chat-Session-Id": chat_session_id,
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.retriever_timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            logger.warning("Retriever API call failed: %s", exc)
            return ""

        results = data.get("results", [])
        if not results:
            logger.info("Retriever returned empty context")
            return ""

        lines: list[str] = []
        for index, item in enumerate(results[: self.context_max_chunks], start=1):
            text = str(item.get("text", "")).strip()
            if len(text) > self.context_max_chars:
                text = text[: self.context_max_chars]
            score = item.get("score", 0.0)
            chunk_id = item.get("chunk_id", "")
            lines.append(f"Context #{index} (chunk_id={chunk_id}, score={score:.4f})\n{text}")
        return "\n\n".join(lines)


def _extract_user_query(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _extract_assistant_response(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return str(message.get("content") or "")
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in reversed(choices):
            message = choice.get("message", {})
            if message.get("role") == "assistant":
                return str(message.get("content") or "")
    return ""


def _extract_trace_id(body: dict[str, Any]) -> str:
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("_langfuse_trace_id") or "")
    return ""


def _extract_chat_session_id(body: dict[str, Any], user: dict[str, Any] | None = None) -> str:
    print("extract_chat_session_id")\
    print(f"body: {body}")
    print(f"user: {user}")
    direct_keys = ("chat_id", "conversation_id", "id", "session_id")
    for key in direct_keys:
        value = body.get(key)
        if value:
            return str(value)
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        for key in direct_keys:
            value = metadata.get(key)
            if value:
                return str(value)
    if isinstance(user, dict):
        value = user.get("id") or user.get("email")
        if value:
            return str(value)
    return "unknown"


def _store_trace_metadata(body: dict[str, Any], *, trace_id: str, chat_session_id: str) -> None:
    metadata = body.get("metadata")
    print(f"store_trace_metadata metadata: {metadata}")
    if not isinstance(metadata, dict):
        metadata = {}
        body["metadata"] = metadata
    metadata["_langfuse_trace_id"] = trace_id
    metadata["_langfuse_chat_session_id"] = chat_session_id


def _history_metrics(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"messages_count": 0, "roles": [], "message_lengths": []}
    roles: list[str] = []
    lengths: list[int] = []
    for message in messages:
        roles.append(str(message.get("role", "unknown")))
        lengths.append(len(str(message.get("content", ""))))
    return {
        "messages_count": len(messages),
        "roles": roles,
        "message_lengths": lengths,
    }
