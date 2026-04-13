FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY stage2_ingestion ./stage2_ingestion
COPY shared ./shared
COPY stage3_1_chunks_indexing ./stage3_1_chunks_indexing

RUN pip install --no-cache-dir \
  "aiokafka>=0.12.0" \
  "chonkie>=1.6.1" \
  "elasticsearch[async]>=8.0.0,<9" \
  "pydantic>=2.11.0" \
  "pydantic-settings>=2.11.0"

CMD ["python", "-m", "stage3_1_chunks_indexing.main"]
