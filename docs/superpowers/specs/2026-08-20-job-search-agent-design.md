# Job Search Agent — Design Spec

**Дата:** 2026-08-20
**Статус:** Утверждено
**Автор:** Дмитрий

## Цель

Личный AI-агент, который ищет работу на произвольных сайтах вакансий (Россия + международные), откликается на подходящие вакансии, пишет сопроводительные письма, отвечает работодателю в чате на сайте и ведёт статистику. Управляется через Telegram свободной перепиской. Архитектура закладывается с расчётом на дальнейший переход к многопользовательскому SaaS.

## Ключевые решения

| Решение | Выбор |
|---|---|
| Стадия | Личный MVP → потом SaaS |
| Язык/стек | Python + LangChain/LangGraph + FastAPI + SQLAlchemy + PostgreSQL |
| Интерфейс | Telegram-бот, свободная переписка (без команд) |
| Взаимодействие с сайтами | Playwright (браузерная автоматизация) |
| Сайты | Россия + международные (MVP — только hh.ru) |
| LLM/VLM | Плавающие провайдеры; дефолт облако, опционально локальный Ollama |
| Векторная БД | PostgreSQL + pgvector |
| Очередь | ARQ + Redis |
| MCP | Клиент (потребление внешних MCP-серверов), post-MVP |

## Архитектура — модульный монолит + agent-first ядро

### Компоненты

| Модуль | Роль | Технологии |
|---|---|---|
| Telegram Gateway | Приём свободных сообщений, уведомления, статистика | aiogram |
| API / Core | Доменная логика, REST (для будущего веб-дашборда и SaaS) | FastAPI, SQLAlchemy async |
| Agent Orchestrator | «Мозг»: стейтфул рабочие циклы | LangGraph |
| Browser Executor | Изолированный «инструмент» с HTTP-интерфейсом, пул Playwright с персистентными профилями | Playwright (отдельный процесс) |
| LLM/VLM Gateway | Плавающие провайдеры (текст + vision), телеметрия | LangChain `BaseChatModel` |
| RAG Service | Индексация CV, эмбеддинги, retrieval | pdfplumber/python-docx + pgvector |
| Site Adapters | Пери-сайтовые адаптеры + generic-адаптер (агент+VLM) | — |
| Task Queue | Длинные циклы поиска/мониторинга | ARQ + Redis |
| DB | Единая БД, в т.ч. векторы | PostgreSQL + pgvector |

Принцип: browser-executor и LLM-gateway — изолированные модули с чётким интерфейсом. Ядро (LangGraph) зависит только от абстракций, не от Playwright/OpenAI напрямую.

### Модель данных

```
users          id, telegram_id, preferences(json), status
profiles       user_id → skills[], experience[], desired_role/salary/location
documents      cv-чанки + embedding (pgvector) + metadata
job_sites      name, adapter_key, credentials(encrypted)
jobs           site, external_id, title, company, url, description, raw(json), status
applications   job_id, status(pending→submitted→viewed→replied→interview→offer→rejected),
               cover_letter, applied_at
conversations  application_id, site, external_thread_id, status, kind(user_chat|employer)
messages       conversation_id, role, content, source, needs_review, sent_at
search_runs    user_id, trigger, filters, applied_count, replied_count
notifications  исходящая очередь в Telegram
llm_calls      provider, model, tokens, cost  (телеметрия)
```

### Агентное ядро (LangGraph)

Верх графа — узел-маршрутизатор (intent router):

```
сообщение из ТГ ──► router (LLM-классификация намерения)
                        ├─ search_job → search → match → decision → apply → monitor → reply → report
                        ├─ stats      → report (откликов N, общений M)
                        ├─ confirm    → продолжение HITL (✅/✏️/❌ на pending-действие)
                        └─ chat       → обычная беседа (LLM, без действий)
```

Намерения: `search_job | stats | confirm | chat`. Сейчас реализован только `search_job`; остальное — беседа.

### Безопасность (эшелонированная защита)

1. **Классификация действий по риску**: `read` → `draft` → `high-risk` (отправка сообщения, отклик).
2. **Human-in-the-loop**: `high-risk` приостанавливает граф (LangGraph `interrupt`) → запрос в ТГ ✅/✏️/❌.
3. **Whitelist инструментов**: фиксированный набор; нет произвольного кода/шелла.
4. **Валидация вывода**: весь LLM-вывод через Pydantic-схему до исполнения.
5. **Контент-гардрейлы**: перед отправкой работодателю — нет утечки секретов, нет выдумок, тон вежливый.
6. **Границы области действия**: browser ходит только по whitelist-доменам.
7. **Rate limiting / анти-бан**: лимит действий/час, рандомизированные паузы.
8. **Секреты**: креды шифруются (Fernet) at rest, не в логах/промптах.
9. **Audit log**: журнал «хотел → одобрено → отменено».

### Browser Executor

HTTP-интерфейс, к которому обращается агент (не импортируется напрямую):

```
navigate(url)            — только whitelist-домены
extract()                — accessibility-tree + упрощённый DOM (основной канал)
screenshot()             — fallback для VLM
click(selector)/type()   — выполнение действия
read_chat()              — входящие сообщения работодателя
send_message(text)       — исходящее (через гардрейлы + HITL)
submit_application()     — отклик + сопроводительное
```

- Персистентные профили: один `user-data-dir` на пользователя.
- Два канала восприятия: DOM/accessibility-tree (LLM) основной, скриншот (VLM) fallback.
- `GenericAdapter` (агент+VLM разбирает незнакомый сайт) + `HhAdapter` (захардкоженные селекторы).

### RAG Service

```
CV (PDF/DOCX) → парсинг → чанкинг по секциям → эмбеддинги → pgvector
```
- Два режима retrieval: семантический по CV + структурированные факты из профиля.
- Материализованный профиль: LLM один раз извлекает структуру при загрузке CV.

### LLM/VLM Gateway

- LangChain `BaseChatModel`, провайдер через конфиг: OpenAI (дефолт) / Ollama (опция).
- Две роли: text-model и vision-model.
- Телеметрия `llm_calls` (provider, model, tokens, cost, latency).

### MCP-клиент (post-MVP)

- `langchain-mcp-adapters` / Python SDK `mcp`.
- Внешние MCP-тулзы → LangChain-инструменты → общий whitelist + policy-слой.
- Allowlist серверов/тулзов из конфига.

## MVP-скоуп

Один сквозной сценарий на hh.ru:
1. Сообщение в ТГ → router → `search_job`
2. Поиск вакансий → кандидаты
3. RAG-match по CV → отбор
4. Отклик + сопроводительное (генерация LLM + RAG, подтверждение в ТГ)
5. Ответ работодателю в чате (draft → ✅)
6. Статистика: откликов N, общений M

Отложено: LinkedIn/международные, GenericAdapter на VLM, полная воронка, веб-дашборд, SaaS/multi-tenant, автопилот, MCP-клиент.

## Тестирование

- Юнит: Pydantic-схемы, policy-слой, гардрейлы.
- LangGraph: граф с fake LLM.
- RAG + DB: testcontainers (Postgres+pgvector).
- Browser: Playwright против локального mock job-сайта.
- Интеграция: сквозной «сообщение ТГ → search_run → отчёт».

## Запуск

```
docker compose up
```
`postgres:pgvector`, `redis`, `app`, `worker`, `browser`, `tg-bot`; `ollama` — опциональный профиль. `.env.example` с инструкцией.
