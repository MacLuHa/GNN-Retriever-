# Stage 2: Ingestion Pipeline

Отдельная реализация этапа 2 из `PROJECT_STAGES.md`.

## Формат выходного сообщения (chunks topic)
```json
{
  "chunk_id": "uuidv7",
  "doc_id": "12",
  "version_id": "v1",
  "title": "Anarchism",
  "page": 1,
  "span_start": 0,
  "span_end": 1024,
  "text": "текст чанка...",
  "metadata": {
    "url": "https://en.wikipedia.org/wiki/Anarchism",
    "title": "Anarchism"
  }
}
```

## Настройка через .env
Скопируйте шаблон:
```bash
cp .env.example .env
```

Ключевые переменные:
- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_CHUNKS_TOPIC`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `HF_DATASET_NAME`
- `HF_DATASET_CONFIG_NAME`
- `HF_DATASET_SPLIT`
- `HF_DATASET_STREAMING`
- `HF_LIMIT`

## Запуск
1. Установить зависимости:
```bash
uv sync
```

2. Запустить ingestion worker:
```bash
uv run stage2-ingestion-kafka
```

Альтернатива через скрипт:
```bash
uv run python scripts/run_stage2_pipeline.py
```

## Режим Hugging Face
Pipeline читает документы из датасета HF и публикует чанки в Kafka.

Для `wikimedia/wikipedia` используется plain text (`text`), поэтому для `source_type=text`
чанкинг идет напрямую по тексту, без PDF/DOCX/HTML-парсеров.

Минимум:
```dotenv
HF_DATASET_NAME=ag_news
HF_DATASET_CONFIG_NAME=
HF_DATASET_SPLIT=train
HF_TEXT_FIELD=text
HF_LIMIT=100
```

Пример для Wikipedia:
```dotenv
HF_DATASET_NAME=wikimedia/wikipedia
HF_DATASET_CONFIG_NAME=20231101.en
HF_DATASET_SPLIT=train[:1%]
HF_DOC_ID_FIELD=id
HF_TITLE_FIELD=title
HF_TEXT_FIELD=text
```
