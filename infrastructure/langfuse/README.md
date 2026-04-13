# Langfuse (Self-Hosted)

Эта директория содержит отдельный стек Docker Compose для Langfuse.

## Сервисы в составе стека

- `langfuse-web`
- `postgres`
- `clickhouse`
- `redis`
- `minio`

## Запуск

1. Скопируйте файл окружения:
   - `cp infrastructure/langfuse/.env.example infrastructure/langfuse/.env`
2. Задайте безопасные значения для:
   - `LANGFUSE_SALT`
   - `LANGFUSE_ENCRYPTION_KEY`
   - `LANGFUSE_NEXTAUTH_SECRET`
3. Запустите стек:
   - `docker compose -f infrastructure/langfuse/docker-compose.yml --env-file infrastructure/langfuse/.env up -d`
4. Откройте:
   - `http://localhost:3001`

## Подключение стека приложения

Установите следующие значения в `services/.env`:

- `LANGFUSE_ENABLED=true`
- `LANGFUSE_HOST=http://host.docker.internal:3001`
- `LANGFUSE_PUBLIC_KEY=<из настроек проекта Langfuse>`
- `LANGFUSE_SECRET_KEY=<из настроек проекта Langfuse>`
