FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# This image builds only retriever API runtime dependencies.
RUN pip install --no-cache-dir \
  "fastapi>=0.116.0" \
  "uvicorn>=0.35.0" \
  "pydantic>=2.11.0" \
  "pydantic-settings>=2.11.0" \
  "elasticsearch[async]==8.17.0" \
  "httpx>=0.28.0" \
  "qdrant-client>=1.15.0" \
  "neo4j>=5.0.0" \
  "torch>=2.2.0" \
  "sentence-transformers>=3.0.0" \
  "langfuse"

COPY stage7_1_retriever ./stage7_1_retriever
COPY stage7_2_reranker ./stage7_2_reranker
COPY stage7_3_retriever_api ./stage7_3_retriever_api

CMD ["python", "-m", "stage7_3_retriever_api.main"]
