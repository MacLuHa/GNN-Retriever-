# NER Benchmark (Ollama vs Hugging Face)

Полностью автономный бенчмарк в отдельной папке `ner_benchmark/`.

## Структура

- `ner_benchmark/ner_benchmark.ipynb` — основной ноутбук
- `ner_benchmark/golden_set/ner_golden.jsonl` — golden set (20 размеченных примеров)
- `ner_benchmark/runs/` — артефакты запусков

## Что сравнивается

1. LLM через Ollama (`/api/chat`)
2. Специализированная NER-модель Hugging Face (`transformers` pipeline)

## Важные метрики

- `entity_level_micro`: precision / recall / f1
- `entity_level_macro`: precision / recall / f1
- `per_label`: precision / recall / f1 + TP/FP/FN для `PER`, `ORG`, `LOC`
- `exact_match_rate` (доля текстов, где набор сущностей совпал полностью)
- `hallucination_rate` (доля лишних сущностей среди предсказаний)
- `miss_rate` (доля пропусков среди gold-сущностей)
- `latency_ms`: avg / p95 / max
- `errors` (сколько текстов не обработалось)

## Формат golden set

Каждая строка в `jsonl`:

```json
{
  "id": "ex_001",
  "text": "Elon Musk visited Berlin to meet engineers from Tesla.",
  "entities": [
    {"text": "Elon Musk", "label": "PER"},
    {"text": "Berlin", "label": "LOC"},
    {"text": "Tesla", "label": "ORG"}
  ]
}
```

## Запуск

Открой ноутбук `ner_benchmark/ner_benchmark.ipynb` и выполни клетки сверху вниз.
В конфиг-клетке можно менять:

- `HF_MODEL_NAME`
- `OLLAMA_MODEL_NAME`
- `RUN_HF` / `RUN_OLLAMA`
- `MAX_SAMPLES`

После запуска результаты сохраняются в `ner_benchmark/runs/<timestamp>/`:

- `summary.json`
- `summary.csv`
- `per_sample_metrics.csv`
