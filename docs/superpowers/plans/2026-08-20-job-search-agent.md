# Job Search Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить личного AI-агента поиска работы: Telegram-бот со свободной перепиской, который ищет вакансии на hh.ru, откликается с сопроводительными, отвечает работодателю в чате и ведёт статистику.

**Architecture:** Модульный монолит (FastAPI) с agent-first ядром на LangGraph. Изолированные модули с HTTP-интерфейсами: Browser Executor (Playwright) и LLM/VLM Gateway (провайдеры через LangChain). Единая PostgreSQL+pgvector, очередь ARQ+Redis, aiogram для Telegram.

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy async, LangChain, LangGraph, Playwright, aiogram, ARQ, PostgreSQL+pgvector, Redis, uv, Docker.

---

## File Structure

```
job-search-agent/
├── pyproject.toml               # uv-проект, зависимости, pytest-конфиг
├── .python-version              # 3.12
├── .env.example                 # шаблон переменных окружения
├── .gitignore / .dockerignore
├── docker-compose.yml           # postgres, redis, app, worker, browser, tg-bot, ollama(optional)
├── Dockerfile                   # мультистейдж builder/runtime
├── Makefile                     # команды: run, test, migrate, lint
├── alembic.ini / migrations/    # alembic-миграции
├── README.md                    # как запустить за 5 минут
├── config.py                    # pydantic-settings (frozen=True)
├── main.py                      # FastAPI app (сервис app: API + lifespan)
│
├── src/
│   ├── database/
│   │   ├── db_settings.py       # async engine (NullPool) + SessionLocal + Base
│   │   └── models.py            # все ORM-модели
│   ├── core/
│   │   ├── security.py          # Fernet-шифрование кредов
│   │   └── audit.py             # журнал действий агента
│   ├── llm/
│   │   ├── gateway.py           # LLMGateway: text/vision, провайдеры, телеметрия
│   │   └── providers.py         # OpenAIProvider, OllamaProvider
│   ├── rag/
│   │   ├── ingest.py            # парсинг CV → чанки → эмбеддинги
│   │   └── retrieve.py          # семантический retrieval + профиль
│   ├── browser/
│   │   ├── server.py            # Playwright HTTP-сервис (процесс browser)
│   │   ├── executor.py          # BrowserExecutor — клиент для агента
│   │   └── adapters.py          # HhAdapter, GenericAdapter
│   ├── agent/
│   │   ├── state.py             # AgentState (Pydantic)
│   │   ├── policy.py            # классификация риска + HITL
│   │   ├── guardrails.py        # контент-гардрейлы исходящих сообщений
│   │   ├── router.py            # intent router
│   │   ├── nodes.py             # search/match/decision/apply/monitor/reply/report
│   │   └── graph.py             # сборка LangGraph
│   ├── stats/
│   │   └── aggregate.py         # агрегация откликов/общений
│   ├── tg/
│   │   ├── bot.py               # aiogram-бот, входная точка tg-bot
│   │   └── handlers.py          # свободная переписка → router
│   ├── mcp/
│   │   └── client.py            # MCP-клиент (post-MVP)
│   └── worker.py                # ARQ worker (длинные циклы)
│
├── telemetry/
│   └── setup_telemetry.py       # OpenTelemetry (опционально)
│
└── tests/
    ├── conftest.py
    ├── mock_site/               # fixture HTML job-сайт для Playwright-тестов
    ├── test_llm.py
    ├── test_rag.py
    ├── test_policy.py
    ├── test_router.py
    ├── test_agent_graph.py
    └── test_stats.py
```

---

## Task 0: Каркас проекта + конфигурация

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.dockerignore`, `config.py`, `Makefile`, `README.md`

- [ ] **Step 1: Инициализировать uv-проект**

```bash
cd /Users/dmitrijpecenkin/job-search-agent
uv init --python 3.12 --name job-search-agent
uv python pin 3.12
```

- [ ] **Step 2: Заполнить `pyproject.toml`**

```toml
[project]
name = "job-search-agent"
version = "0.1.0"
description = "AI-агент поиска работы"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0,<0.116.0",
    "uvicorn[standard]>=0.30.0,<0.31.0",
    "sqlalchemy[asyncio]>=2.0.40,<2.1.0",
    "asyncpg>=0.30.0,<0.31.0",
    "alembic>=1.13.0,<2.0.0",
    "pydantic-settings>=2.10.0,<2.11.0",
    "httpx>=0.28.0,<0.29.0",
    "langchain>=0.3.0,<0.4.0",
    "langchain-openai>=0.2.0,<0.3.0",
    "langgraph>=0.2.0,<0.3.0",
    "playwright>=1.48.0,<2.0.0",
    "aiogram>=3.13.0,<4.0.0",
    "arq>=0.26.0,<0.27.0",
    "redis>=5.0.0,<6.0.0",
    "pdfplumber>=0.11.0,<0.12.0",
    "python-docx>=1.1.0,<2.0.0",
    "pgvector>=0.3.0,<0.4.0",
    "cryptography>=43.0.0,<44.0.0",
]

[project.optional-dependencies]
dev = [
    "black>=24.0.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.25.0",
    "testcontainers[postgres]>=4.0.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Установить зависимости**

```bash
uv sync
```

- [ ] **Step 4: Создать `config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения. Значения берутся из .env или окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        frozen=True,
    )

    # App
    DEBUG: bool = False

    # DB
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "jobagent"
    DB_PASS: str = "jobagent"
    DB_NAME: str = "jobagent"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # LLM
    LLM_PROVIDER: str = "openai"  # openai | ollama
    LLM_TEXT_MODEL: str = "gpt-4o-mini"
    LLM_VISION_MODEL: str = "gpt-4o"
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Browser executor
    BROWSER_EXECUTOR_URL: str = "http://localhost:8010"

    # Telegram
    TG_BOT_TOKEN: str = ""

    # Security
    FERNET_KEY: str = ""  # base64-encoded 32-byte key


settings = Settings()
```

- [ ] **Step 5: Создать `.env.example`, `.gitignore`, `.dockerignore`, `Makefile`, `README.md`**

`.env.example`:
```bash
DEBUG=true
DB_HOST=localhost
DB_PORT=5432
DB_USER=jobagent
DB_PASS=jobagent
DB_NAME=jobagent
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
TG_BOT_TOKEN=123456:ABC...
FERNET_KEY=
```

`.gitignore`:
```
.env
.venv
__pycache__/
*.pyc
.pytest_cache/
.uv/
```

`.dockerignore`:
```
.venv
__pycache__
*.pyc
.git
.gitignore
.env
```

`Makefile`:
```makefile
run:
	uv run uvicorn main:app --reload

test:
	uv run pytest

lint:
	uv run black --check src tests config.py main.py

format:
	uv run black src tests config.py main.py

migrate:
	uv run alembic upgrade head
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: project scaffold + config"
```

---

## Task 1: Модель данных + alembic

**Files:**
- Create: `src/database/db_settings.py`, `src/database/models.py`
- Create: `alembic.ini`, `migrations/env.py`

- [ ] **Step 1: `src/database/db_settings.py`** (engine NullPool, SessionLocal, Base)

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool

from config import settings


class Base(DeclarativeBase):
    """База для всех ORM-моделей."""


sql_link = (
    f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

engine = create_async_engine(sql_link, echo=False, poolclass=NullPool)
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, class_=AsyncSession, bind=engine
)
```

- [ ] **Step 2: `src/database/models.py`** — все модели из спеки

```python
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from src.database.db_settings import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="active")


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    skills: Mapped[list] = mapped_column(JSONB, default=list)
    experience: Mapped[list] = mapped_column(JSONB, default=list)
    desired_role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    desired_salary: Mapped[str | None] = mapped_column(String(128), nullable=True)
    desired_location: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(1536), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class JobSite(Base):
    __tablename__ = "job_sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    adapter_key: Mapped[str] = mapped_column(String(64))
    credentials: Mapped[str] = mapped_column(Text, default="")  # Fernet-encrypted


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("job_sites.id"))
    external_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(512))
    company: Mapped[str | None] = mapped_column(String(512), nullable=True)
    url: Mapped[str] = mapped_column(String(1024))
    description: Mapped[str] = mapped_column(Text, default="")
    raw: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="new")


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    cover_letter: Mapped[str] = mapped_column(Text, default="")
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id"), nullable=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    site_id: Mapped[int | None] = mapped_column(ForeignKey("job_sites.id"), nullable=True)
    external_thread_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), default="employer")  # employer | user_chat
    status: Mapped[str] = mapped_column(String(32), default="active")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="agent")  # agent | user | employer
    needs_review: Mapped[bool] = mapped_column(default=False)
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SearchRun(Base):
    __tablename__ = "search_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    trigger: Mapped[str] = mapped_column(String(64))
    filters: Mapped[dict] = mapped_column(JSONB, default=dict)
    applied_count: Mapped[int] = mapped_column(default=0)
    replied_count: Mapped[int] = mapped_column(default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    sent: Mapped[bool] = mapped_column(default=False)


class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    tokens: Mapped[int] = mapped_column(default=0)
    cost: Mapped[float] = mapped_column(default=0.0)
    latency_ms: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
```

- [ ] **Step 3: Настроить alembic** (`alembic init migrations`, обновить `migrations/env.py` на async engine, `migrations/script.py.mako` для автогенерации)

- [ ] **Step 4: Сгенерировать и применить миграцию**

```bash
uv run alembic revision --autogenerate -m "init models"
uv run alembic upgrade head
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: database models + alembic"
```

---

## Task 2: LLM/VLM Gateway

**Files:**
- Create: `src/llm/gateway.py`, `src/llm/providers.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Провайдеры** — `src/llm/providers.py`

```python
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def build_text_model(provider: str, model: str, api_key: str, base_url: str = "") -> BaseChatModel:
    """Создать текстовую модель по имени провайдера."""
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
    if provider == "ollama":
        from langchain_ollama import ChatOllama  # локальный импорт

        return ChatOllama(model=model, base_url=base_url)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def build_vision_model(provider: str, model: str, api_key: str, base_url: str = "") -> BaseChatModel:
    """Создать vision-модель."""
    return build_text_model(provider, model, api_key, base_url)
```

- [ ] **Step 2: Gateway** — `src/llm/gateway.py` (обёртка с телеметрией)

```python
from typing import Any

from langchain_core.language_models import BaseChatModel

from config import settings


class LLMGateway:
    """Единая точка доступа к текстовым и vision-моделям с телеметрией."""

    def __init__(self) -> None:
        self.text_model = self._build_text()
        self.vision_model = self._build_vision()

    def _build_text(self) -> BaseChatModel:
        from src.llm.providers import build_text_model

        return build_text_model(
            settings.LLM_PROVIDER,
            settings.LLM_TEXT_MODEL,
            settings.OPENAI_API_KEY,
            settings.OLLAMA_BASE_URL,
        )

    def _build_vision(self) -> BaseChatModel:
        from src.llm.providers import build_vision_model

        return build_vision_model(
            settings.LLM_PROVIDER,
            settings.LLM_VISION_MODEL,
            settings.OPENAI_API_KEY,
            settings.OLLAMA_BASE_URL,
        )

    async def invoke_structured(self, model: BaseChatModel, messages: list, schema: type) -> Any:
        """Вызвать модель с Pydantic-валидацией вывода."""
        structured = model.with_structured_output(schema)
        return await structured.ainvoke(messages)
```

- [ ] **Step 3: Тест с fake-провайдером** — `tests/test_llm.py` (проверяет выбор провайдера и падение на неизвестном)

```python
import pytest

from src.llm.providers import build_text_model


def test_build_text_model_unknown_provider():
    with pytest.raises(ValueError):
        build_text_model("unknown", "m", "k", "")
```

- [ ] **Step 4: Запустить тест**

```bash
uv run pytest tests/test_llm.py -v
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: llm gateway with pluggable providers"
```

---

## Task 3: RAG Service

**Files:**
- Create: `src/rag/ingest.py`, `src/rag/retrieve.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: `src/rag/ingest.py`** — парсинг CV (PDF/DOCX) → чанки

```python
from pathlib import Path


async def extract_cv_text(path: Path) -> str:
    """Извлечь текст из CV (PDF или DOCX)."""
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if path.suffix.lower() == ".docx":
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


def chunk_by_sections(text: str, max_chars: int = 1500) -> list[str]:
    """Разбить текст CV на чанки по пустым строкам/секциям."""
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if not line.strip() and current:
            if len(current) > max_chars:
                chunks.extend(_split_long(current, max_chars))
            else:
                chunks.append(current)
            current = ""
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current)
    return [c for c in chunks if c.strip()]
```

- [ ] **Step 2: `src/rag/retrieve.py`** — эмбеддинги + поиск по pgvector + извлечение профиля

- [ ] **Step 3: Тест на `chunk_by_sections` и `extract_cv_text` (fixture docx)**

- [ ] **Step 4: Запустить тест, Commit**

---

## Task 4: Browser Executor

**Files:**
- Create: `src/browser/server.py` (Playwright HTTP-сервис), `src/browser/executor.py`, `src/browser/adapters.py`
- Create: `tests/mock_site/` (fixture HTML)
- Test: `tests/test_browser.py`

- [ ] **Step 1: `src/browser/executor.py`** — HTTP-клиент (httpx), интерфейс для агента:

```python
class BrowserExecutor:
    """Клиент к browser-сервису. Агент зависит только от этого интерфейса."""

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def extract(self, user_id: int) -> dict:
        """Получить accessibility-tree текущей страницы."""
        return (await self._client.post("/extract", json={"user_id": user_id})).json()

    async def screenshot(self, user_id: int) -> bytes:
        return (await self._client.post("/screenshot", json={"user_id": user_id})).content

    async def navigate(self, user_id: int, url: str) -> dict: ...
    async def click(self, user_id: int, selector: str) -> dict: ...
    async def type_text(self, user_id: int, selector: str, text: str) -> dict: ...
    async def send_message(self, user_id: int, text: str) -> dict: ...
```

- [ ] **Step 2: `src/browser/server.py`** — FastAPI-приложение с Playwright-пулом и персистентными профилями (`user-data-dir` на пользователя), whitelist доменов, анти-бан паузы.

- [ ] **Step 3: `src/browser/adapters.py`** — `HhAdapter` (захардкоженные селекторы hh.ru) + `GenericAdapter` (DOM + VLM fallback).

- [ ] **Step 4: `tests/mock_site/`** — мини-сайт с вакансией и чатом для Playwright-тестов (без реальных сайтов).

- [ ] **Step 5: Commit**

---

## Task 5: Агентное ядро (LangGraph) + безопасность

**Files:**
- Create: `src/agent/state.py`, `src/agent/policy.py`, `src/agent/guardrails.py`, `src/agent/router.py`, `src/agent/nodes.py`, `src/agent/graph.py`
- Test: `tests/test_policy.py`, `tests/test_router.py`, `tests/test_agent_graph.py`

- [ ] **Step 1: `src/agent/state.py`** — AgentState (TypedDict/Pydantic)

```python
class AgentState(TypedDict):
    user_id: int
    intent: str
    messages: list[dict]
    candidates: list[dict]
    decisions: list[dict]
    pending_action: dict | None
    report: dict | None
    needs_human: bool
```

- [ ] **Step 2: `src/agent/policy.py`** — классификация риска + HITL

```python
RISK_LEVELS = {"read": 0, "draft": 1, "high_risk": 2}

def classify_action(action: str) -> str:
    """Определить уровень риска действия."""
    if action in {"search", "extract", "read_chat", "screenshot"}:
        return "read"
    if action in {"draft_cover_letter", "draft_reply"}:
        return "draft"
    if action in {"apply", "send_message"}:
        return "high_risk"
    return "read"
```

- [ ] **Step 3: `src/agent/guardrails.py`** — проверка исходящих сообщений (нет секретов, нет выдумок, Pydantic-валидация).

- [ ] **Step 4: `src/agent/router.py`** — классификация намерения (`search_job | stats | confirm | chat`) с Pydantic-выходом.

- [ ] **Step 5: `src/agent/nodes.py`** — узлы: search, match, decision, apply, monitor, reply, report.

- [ ] **Step 6: `src/agent/graph.py`** — сборка графа с `interrupt` для HITL.

- [ ] **Step 7: Тесты графа с fake LLM** (проверка переходов, HITL-прерываний).

- [ ] **Step 8: Commit**

---

## Task 6: Telegram Gateway

**Files:**
- Create: `src/tg/bot.py`, `src/tg/handlers.py`

- [ ] **Step 1: `src/tg/handlers.py`** — любой текст → router → диспетчеризация. `confirm`-ответы имеют приоритет (проверка pending_action).

- [ ] **Step 2: `src/tg/bot.py`** — aiogram-бот (polling для MVP).

- [ ] **Step 3: Commit**

---

## Task 7: Статистика

**Files:**
- Create: `src/stats/aggregate.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: `aggregate.py`** — подсчёт `applied_count`/`replied_count` из `applications`/`conversations` за `search_run`.

- [ ] **Step 2: Тест, Commit**

---

## Task 8: Docker, worker, интеграция, README

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `src/worker.py`, `main.py`

- [ ] **Step 1: `main.py`** — FastAPI app (lifespan: создание клиентов), healthcheck.

- [ ] **Step 2: `src/worker.py`** — ARQ worker с длинными job-функциями (search_run).

- [ ] **Step 3: `Dockerfile`** — мультистейдж builder/runtime.

- [ ] **Step 4: `docker-compose.yml`** — postgres(pgvector), redis, app, worker, browser, tg-bot, ollama(optional profile).

- [ ] **Step 5: Сквозной тест + README («запуск за 5 минут»).**

- [ ] **Step 6: Commit**

---

## Task 9: MCP-клиент (post-MVP)

**Files:**
- Create: `src/mcp/client.py`

- [ ] **Step 1:** `langchain-mcp-adapters` — загрузка внешних MCP-серверов в инструменты агента, allowlist из конфига, интеграция с policy-слоем.

- [ ] **Step 2: Commit**

---

## Self-Review Notes

- Спека покрыта: Task 1 (модель данных), Task 2 (LLM gateway), Task 3 (RAG), Task 4 (browser), Task 5 (ядро+безопасность), Task 6 (TG), Task 7 (статистика), Task 8 (docker/интеграция), Task 9 (MCP).
- Безопасность (9 уровней) реализуется в Task 5 (policy/guardrails/HITL) + Task 4 (whitelist доменов, rate limit) + Task 0 (Fernet).
- Placeholder-скан: задачи 3–8 содержат сигнатуры интерфейсов, полный код заполняется при реализации (TDD).
