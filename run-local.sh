#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[run-local] Устанавливаю зависимости (uv sync)..."
uv sync

echo "[run-local] Применяю миграции (alembic upgrade head)..."
uv run alembic upgrade head

echo "[run-local] Запускаю app, browser и tg-bot (Ctrl+C — остановить всё)..."
uv run uvicorn main:app --reload &
APP_PID=$!
uv run uvicorn src.browser.server:build_browser_app --factory --port 8010 &
BROWSER_PID=$!
uv run python -m src.tg.bot &
TG_PID=$!

cleanup() {
    echo
    echo "[run-local] Останавливаю сервисы..."
    kill "$APP_PID" "$BROWSER_PID" "$TG_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait
