# Сервисы: Open WebUI + Retriever

В этой папке находится runtime-конфигурация Compose для UI и слоя retrieval.

## Сервисы

- `stage3-1-chunks-indexing` — индексирует текст чанков и публикует события векторизации.
- `stage3-2-chunks-vectorizing` — генерирует эмбеддинги и сохраняет векторы в Qdrant.
- `stage3-3-graph-knowledge-base` — извлекает сущности/отношения графа и обновляет Neo4j.
- `open-webui` — чат-интерфейс.
- `openwebui-pipelines` — runtime пайплайнов для retrieval-фильтра.
- `stage7-3-retriever-api` — HTTP API-обёртка над `stage7_1_retriever`.

Ожидается, что сервисы слоя данных (`Elasticsearch`, `Qdrant`, `Neo4j`, `Ollama`, `Kafka`) доступны по URL из `deploy/services/.env`.

## Первый запуск

1. Скопируйте env-файл:
   - `cp deploy/services/.env.example deploy/services/.env`
2. Запустите сервисы (из корня репозитория):
   - `docker compose -f deploy/services/docker-compose.yml --env-file deploy/services/.env up -d --build`
3. Проверьте состояние:
   - `curl http://localhost:8010/health`
   - откройте `http://localhost:3000`

## Submodule Open WebUI

Open WebUI подключён как git submodule в `vendor/open-webui`.

- Инициализация после `clone`:
  - `git submodule update --init --recursive`
- Обновление до последнего коммита отслеживаемой ветки:
  - `git submodule update --remote --recursive`

### Переключение submodule на ваш форк

Если нужно использовать ваш форк как `origin`:

1. Установите URL форка:
   - `git -C vendor/open-webui remote set-url origin <YOUR_FORK_URL>`
2. Выполните `fetch` и `checkout` вашей ветки:
   - `git -C vendor/open-webui fetch origin`
   - `git -C vendor/open-webui checkout <YOUR_BRANCH>`
3. Закоммитьте обновлённый указатель submodule в основном репозитории.

## Настройка pipeline в Open WebUI

Файл pipeline расположен в:
- `deploy/services/openwebui/pipelines/retriever_pipeline.py`

В админ-панели Open WebUI:
1. Перейдите в `Admin Panel -> Pipelines`.
2. Убедитесь, что URL провайдера указывает на `http://openwebui-pipelines:9099`.
3. Загрузите `retriever_pipeline.py`.
4. Подключите filter pipeline к нужной модели.

## Интеграция с Langfuse

- Запустите стек Langfuse отдельно:
  - `docker compose -f deploy/infrastructure/langfuse/docker-compose.yml --env-file deploy/infrastructure/langfuse/.env up -d`
- Включите переменные в `deploy/services/.env`:
  - `LANGFUSE_ENABLED=true`
  - `LANGFUSE_HOST=http://host.docker.internal:3001`
  - `LANGFUSE_PUBLIC_KEY=<key>`
  - `LANGFUSE_SECRET_KEY=<secret>`

## Примечания

- Логи по дизайну остаются на английском.
- Таймаут retriever и лимиты контекста управляются переменными в `deploy/services/.env`.
- Cross-encoder reranking настраивается через переменные `RERANK_*` и включается `RERANK_ENABLED=true`.
- Если retriever недоступен, pipeline автоматически переходит в fallback-режим без RAG-контекста.
