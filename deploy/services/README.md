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

## Интеграция `retriever_pipeline` с Open WebUI

### Как это устроено в `docker-compose.yml`

- Контейнер **`openwebui-pipelines`** (образ `ghcr.io/open-webui/pipelines`) монтирует `./openwebui/pipelines` в **`/app/pipelines`**.
- Сервер пайплайнов при старте подхватывает **только файлы `*.py` в корне** `/app/pipelines` (вложенные каталоги с другим `.py` **не** сканируются).
- Файл фильтра: [`openwebui/pipelines/retriever_pipeline.py`](openwebui/pipelines/retriever_pipeline.py). Рядом создаётся каталог [`openwebui/pipelines/retriever_pipeline/valves.json`](openwebui/pipelines/retriever_pipeline/valves.json) (настройки valves; пустой `{}` — дефолты из кода).
- **`open-webui`** получает `OPENAI_API_BASE_URLS=http://openwebui-pipelines:9099` и `OPENAI_API_KEYS=${OPENWEBUI_PIPELINES_API_KEY}` — отдельное «OpenAI»-подключение на сервис пайплайнов (нужно для списка моделей/маршрутизации в актуальных версиях UI).
- Контейнер **`openwebui-pipelines`** должен видеть API ретривера: `RETRIEVER_API_URL` (в `.env` по умолчанию `http://stage7-3-retriever-api:8010`).

### Переменные в `deploy/services/.env`

См. `.env.example`: `OPENWEBUI_PIPELINES_API_KEY`, `OPENWEBUI_PIPELINES_PORT`, `RETRIEVER_API_URL`, `RETRIEVER_TIMEOUT_SEC`, `RETRIEVER_TOP_K`, `RETRIEVER_MAX_CONTEXT_CHARS`, `RETRIEVER_MAX_CONTEXT_CHUNKS`, при необходимости `LANGFUSE_*`.

### Шаги в UI Open WebUI

Точные названия меню могут отличаться по версии; логика такая:

1. Запустите стек: `openwebui-pipelines` и `open-webui` должны быть в одной Docker-сети (как в compose).
2. Убедитесь, что в логах `openwebui-pipelines` есть строка вида `Loaded module: retriever_pipeline` (если нет — проверьте, что **`retriever_pipeline.py` лежит в корне** `openwebui/pipelines/`, и перезапустите контейнер).
3. **Admin Panel → Settings → Connections (или Pipelines):** укажите URL сервиса пайплайнов и API-ключ:
   - URL: из браузера на хосте — `http://localhost:<OPENWEBUI_PIPELINES_PORT>` (например `9099`); внутри compose WebUI уже смотрит на `http://openwebui-pipelines:9099`.
   - Ключ: тот же, что **`OPENWEBUI_PIPELINES_API_KEY`** и **`PIPELINES_API_KEY`** у контейнера `openwebui-pipelines`.
4. Filter **Retriever RAG** (`retriever_pipeline`) должен появиться в списке пайплайнов; в valves по умолчанию **`pipelines: ["*"]`** — фильтр цепляется ко всем моделям. При необходимости сузьте список id моделей в UI или в `valves.json`.
5. Для ответов чата выберите модель из **Ollama** (`OLLAMA_BASE_URL` в `.env` у `open-webui`). Filter перед запросом к модели вызывает `POST {RETRIEVER_API_URL}/retrieve` и дописывает контекст в system message.

Проверка API пайплайнов с хоста: `curl -s -H "Authorization: Bearer <OPENWEBUI_PIPELINES_API_KEY>" http://localhost:9099/v1/models`.

## Интеграция с Langfuse

- Запустите стек Langfuse отдельно:
  - `docker compose -f deploy/infrastructure/langfuse/docker-compose.yml --env-file deploy/infrastructure/langfuse/.env up -d`
- Включите переменные в `deploy/services/.env`:
  - `LANGFUSE_ENABLED=true` (без этого клиент SDK не создаётся)
  - `LANGFUSE_HOST=http://host.docker.internal:3001` (порт как в `LANGFUSE_PORT` инфраструктуры; внутри контейнера Langfuse слушает `3000`, снаружи проброшен на хост)
  - `LANGFUSE_PUBLIC_KEY=<key>`
  - `LANGFUSE_SECRET_KEY=<secret>` (ключи из того же проекта в UI Langfuse)

### Если трейсы не появляются в UI

1. В логах `stage7-3-retriever-api` после старта должно быть `Langfuse tracing active host=...` либо предупреждение о неактивном клиенте.
2. **Retriever API** после каждого `POST /retrieve` вызывает `flush()` у SDK — события не должны «застревать» только в буфере.
3. На Linux у контейнеров **`openwebui-pipelines`**, **`stage7-3-retriever-api`** и **`open-webui`** в compose задан `extra_hosts: host.docker.internal:host-gateway`, иначе URL `http://host.docker.internal:...` из контейнера не откроется.
4. Проверка сети из контейнера:  
   `docker exec -it openwebui-pipelines wget -qO- --timeout=3 http://host.docker.internal:3001/api/public/health` (или ваш `LANGFUSE_HOST` без лишнего пути).
5. Пайплайн `retriever_pipeline`: после каждого `inlet` вызывается `flush()`; сопоставление inlet→`outlet` идёт по `chat_id` / `session_id` / `conversation_id` / `_langfuse_trace_id` (поле `id` в теле запроса **не** используется — в `outlet` это id сообщения). При старте контейнера выполняется `auth_check` к Langfuse; при неверных ключах будет **WARNING** в логах. Переменная `LANGFUSE_TRACING_ENVIRONMENT` должна быть в нижнем регистре и соответствовать `[a-z0-9-_]+` без префикса `langfuse`, иначе сервер отклоняет события — пайплайн подставит `pipeline`. Для отладки: `LANGFUSE_DEBUG=true`, `LANGFUSE_LOG_TRACE_URL=true` (в лог попадёт URL трейса). Образ ставит `langfuse` 2.x — пересоберите контейнер после изменения Dockerfile.

## Примечания

- Логи по дизайну остаются на английском.
- Таймаут retriever и лимиты контекста управляются переменными в `deploy/services/.env`.
- Cross-encoder reranking настраивается через переменные `RERANK_*` и включается `RERANK_ENABLED=true`.
- Если retriever недоступен, pipeline автоматически переходит в fallback-режим без RAG-контекста.
