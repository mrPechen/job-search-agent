# Job Search Agent

Личный AI-агент поиска работы: Telegram-бот со свободной перепиской, который ищет
вакансии на hh.ru, откликается с сопроводительными письмами, отвечает работодателю
в чате и ведёт статистику.

## Архитектура

Сервисы, поднимаемые через Docker Compose:

- `postgres` — база данных (PostgreSQL 16 + pgvector для эмбеддингов)
- `redis` — очередь задач и кэш
- `app` — FastAPI API (`main.py`), healthcheck на `/health`
- `browser` — Playwright-сервис с Chromium, управляет браузерными профилями
- `tg-bot` — Telegram-бот (aiogram)
- `worker` — фоновый обработчик задач (ARQ)
- `ollama` — локальные LLM (опционально, профиль `local`)

## Требования

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- Для локальной разработки без Docker: Python 3.12+ и [uv](https://docs.astral.sh/uv/)

## Быстрый старт (Docker)

```bash
cp .env.example .env
# открой .env и заполни OPENAI_API_KEY и TG_BOT_TOKEN
docker compose up -d postgres
docker compose run --rm app alembic upgrade head
docker compose up -d
```

После старта проверь живость API:

```bash
curl http://localhost:8000/health   # {"ok":true}
```

Хосты `postgres`, `redis` и `browser` внутри сети Compose уже подставлены в
`docker-compose.yml` через `environment`, поэтому один и тот же `.env` подходит
и для Docker, и для локальной разработки.

### Как пользоваться ботом

Открой своего бота в Telegram и напиши:

> посмотри что нового по работе

Бот разберёт намерение, найдёт вакансии на hh.ru через browser-сервис, оценит их
релевантность и при необходимости спросит подтверждение перед откликом.

## Локальная разработка (без Docker)

Нужны запущенные PostgreSQL и Redis (например, `docker compose up -d postgres redis`).

```bash
cp .env.example .env
# заполни OPENAI_API_KEY и TG_BOT_TOKEN, или настрой Ollama (см. ниже)
uv sync
uv run alembic upgrade head

# в отдельных терминалах:
uv run uvicorn main:app --reload
uv run uvicorn src.browser.server:build_browser_app --factory --port 8010
uv run python -m src.tg.bot
```

Запуск тестов:

```bash
uv run pytest -q
```

## Локальные LLM через Ollama (без OpenAI)

Агент умеет работать полностью локально: текстовая модель, vision-модель и
эмбеддинги для RAG — всё через Ollama.

### 1. Установка Ollama

macOS (Homebrew) или установщик с [ollama.com](https://ollama.com):

```bash
brew install ollama
ollama --version
```

### 2. Запуск и включение API

Ollama сразу поднимает HTTP API на `http://localhost:11434` — отдельно включать
API не нужно, он включён по умолчанию.

- **macOS-приложение** (с ollama.com): запускается как фоновый сервис, API слушает
  `localhost:11434`.
- **Через CLI**: `ollama serve`

Проверка, что API жив:

```bash
curl http://localhost:11434/api/tags
```

Если API нужен не только с localhost (например, для агента в Docker-контейнере
или с другой машины), открой его на всех интерфейсах:

```bash
# CLI:
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# macOS-приложение (затем перезапусти приложение):
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

Тогда агент из Docker достучится по `http://host.docker.internal:11434`.

### 3. Какие модели установить

```bash
ollama pull llama3.2          # текстовая модель: интент, скоринг, письма
ollama pull qwen3-vl:4b       # vision-модель для разбора страниц
ollama pull nomic-embed-text  # эмбеддинги для RAG (768 измерений)
```

- Текстовая модель обязана поддерживать structured output (JSON) — `llama3.2`
  подходит. Альтернативы: `qwen2.5:7b`, `gemma3:4b`.
- Vision-модель (`LLM_VISION_MODEL`) реально используется агентом: она «смотрит»
  на страницы и управляет браузером при поиске и отклике на любых сайтах.
  Для Ollama подойдёт `qwen3-vl:4b` (или `llama3.2-vision`), для облака — `gpt-4o`.
  При первом общении бот спросит, на каких сайтах искать, и запомнит список для
  каждого пользователя.

Проверка установленных моделей: `ollama list`.

### 4. Настройка `.env`

```bash
LLM_PROVIDER=ollama
LLM_TEXT_MODEL=llama3.2
LLM_VISION_MODEL=llama3.2-vision
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

Размерность `EMBEDDING_DIM` обязана совпадать с моделью эмбеддингов: у
`nomic-embed-text` — 768, у `mxbai-embed-large` — 1024, у OpenAI
`text-embedding-3-small` — 1536. После смены `EMBEDDING_DIM` пересоздай колонку
(проще всего сбросить БД: `docker compose down -v` + заново `alembic upgrade head`),
т.к. pgvector хранит фиксированную размерность.

### 5. Память и производительность

```bash
# держать модель загруженной в память (вместо выгрузки после ответа):
OLLAMA_KEEP_ALIVE=-1

# сколько моделей держать загруженными одновременно (по умолчанию 1):
OLLAMA_MAX_LOADED_MODELS=2

# ускорить инференс через flash attention (при поддержке GPU):
OLLAMA_FLASH_ATTENTION=1
```

Загруженные сейчас модели: `ollama ps`.

## Использовать свой Chrome с текущей сессией (CDP)

По умолчанию browser-сервис запускает собственный Chromium Playwright с отдельным
профилем. Чтобы агент работал **в твоём реальном Chrome** (с твоими логинами,
например уже выполненным входом на hh.ru), используй режим CDP:

1. Полностью закрой Chrome (иначе флаг отладки не подхватится).
2. Запусти Chrome с портом отладки:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
```

3. В `.env` укажи:

```bash
BROWSER_MODE=cdp
BROWSER_CDP_URL=http://localhost:9222
```

4. Запусти browser-сервис **локально** (не в Docker — в контейнере нет доступа к
   твоему Chrome):

```bash
uv run uvicorn src.browser.server:build_browser_app --factory --host 0.0.0.0 --port 8010
```

Теперь агент ходит по тому же Chrome, где ты залогинен, и видит твою текущую
сессию. Важно: режим CDP работает только с Chrome/Chromium/Edge — Safari не
поддерживает CDP.

## Безопасность browser-сервиса

Browser-сервис управляет браузером (в CDP-режиме — твоим реальным Chrome с
логинами). Агент ограничивает навигацию списком сайтов конкретного пользователя,
но сам сервис по умолчанию слушает без аутентификации. Для запуска вне localhost
обязательно задай общий секрет в `.env` — агент и browser-сервис используют один
и тот же токен:

```bash
BROWSER_API_TOKEN=случайная-длинная-строка
```

Без этого токена любой, кто достучится до порта `8010`, сможет управлять
браузером. Не публикуй порт `8010` наружу и не запускай browser-сервис в CDP-режиме
на машине, доступной из сети, без установленного `BROWSER_API_TOKEN`.

