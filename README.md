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
# заполни OPENAI_API_KEY и TG_BOT_TOKEN
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

## Локальные LLM (опционально)

Для работы без OpenAI API подними Ollama и выбери модели в `.env`:

```bash
docker compose --profile local up -d ollama
ollama pull llama3.2
```

В `.env` укажи `LLM_PROVIDER=ollama` и `OLLAMA_BASE_URL=http://localhost:11434`
(при запуске агента внутри Docker используй `http://host.docker.internal:11434`).

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

