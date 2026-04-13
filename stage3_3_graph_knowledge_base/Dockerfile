FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY shared ./shared
COPY stage3_3_graph_knowledge_base ./stage3_3_graph_knowledge_base

RUN pip install --no-cache-dir \
  "aiokafka>=0.12.0" \
  "elasticsearch[async]>=8.0.0,<9" \
  "httpx>=0.28.0" \
  "neo4j>=5.0.0" \
  "qdrant-client>=1.15.0" \
  "simplemma>=1.1.0" \
  "pydantic>=2.11.0" \
  "pydantic-settings>=2.11.0" \
  "gliner>=0.2.24"

CMD ["python", "-m", "stage3_3_graph_knowledge_base.main"]
