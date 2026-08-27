#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[run-docker] Поднимаю postgres..."
docker compose up -d postgres

echo "[run-docker] Жду готовности postgres..."
until docker compose exec -T postgres pg_isready -U jobagent -d jobagent >/dev/null 2>&1; do
    sleep 1
done

echo "[run-docker] Применяю миграции..."
docker compose run --rm app alembic upgrade head

echo "[run-docker] Поднимаю все сервисы..."
docker compose up -d

echo "[run-docker] Готово. Проверка: curl http://localhost:8000/health"
