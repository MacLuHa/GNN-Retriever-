FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY shared ./shared
COPY stage3_2_chunks_vectorizing ./stage3_2_chunks_vectorizing

RUN pip install --no-cache-dir \
  "aiokafka>=0.12.0" \
  "httpx>=0.28.0" \
  "qdrant-client>=1.15.0" \
  "pydantic>=2.11.0" \
  "pydantic-settings>=2.11.0"

CMD ["python", "-m", "stage3_2_chunks_vectorizing.main"]
