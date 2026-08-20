# Job Search Agent

Личный AI-агент поиска работы: Telegram-бот со свободной перепиской, который ищет
вакансии на hh.ru, откликается с сопроводительными письмами, отвечает работодателю
в чате и ведёт статистику.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (для PostgreSQL + Redis)

## Быстрый старт

```bash
cp .env.example .env
# заполни OPENAI_API_KEY и TG_BOT_TOKEN в .env
uv sync
uv run uvicorn main:app --reload
```
