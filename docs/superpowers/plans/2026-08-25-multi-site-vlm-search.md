# Multi-Site VLM Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Поддержать поиск и авто-отклик на произвольных сайтах по работе: onboarding (юзер называет сайты), per-user whitelist доменов, универсальный разбор страниц через VLM + Playwright.

**Architecture:** Единый браузерный цикл на VLM (`BrowserLoop`) для всех сайтов, включая hh.ru. `UniversalSearcher`/`UniversalApplier` заменяют hh-специфичные `HhSearcher`/`submit_application`. Onboarding — pre-graph шаг в `TgService`. Работа с Ollama и OpenAI через существующий `LLMGateway`.

**Tech Stack:** Python 3.12+, LangGraph, LangChain (ChatOllama/ChatOpenAI), Playwright, SQLAlchemy async, Alembic, pytest, uv.

---

## File Structure

Изменения относительно текущего кода:

```
src/
├── agent/
│   ├── sites.py            # NEW: Sites schema, normalize_domain, extract_domains,
│   │                       #      parse_sites_message, SitesRepository
│   ├── browser_loop.py     # NEW: BrowserAction/SearchOutcome/ApplyOutcome, BrowserLoop
│   ├── searcher.py         # NEW: UniversalSearcher, UniversalApplier
│   ├── graph.py            # без изменений
│   ├── router.py           # без изменений
├── browser/
│   ├── server.py           # MOD: allowed_domains, scroll, back; удалить /submit
│   ├── executor.py         # MOD: navigate(allowed_domains), scroll, back; удалить submit_application
│   ├── searcher.py         # DEL: HhSearcher
│   └── adapters.py         # DEL: HhAdapter/GenericAdapter (chat-селекторы инлайним в server.py)
├── database/
│   └── models.py           # MOD: UserSite модель + UniqueConstraint import
├── tg/
│   ├── service.py          # MOD: onboarding (gateway + sites_repo)
│   └── bot.py              # MOD: composition root (UniversalSearcher/Applier)
config.py                   # MOD: BROWSER_MAX_STEPS
migrations/versions/        # NEW: add_user_sites
tests/
├── test_domains.py         # NEW
├── test_browser_loop.py    # NEW
├── test_searcher.py        # NEW
├── test_browser.py         # MOD: whitelist/scroll/back; удалить test_submit_application
└── test_tg.py              # MOD: sites_repo/gateway в _build_service + onboarding-тесты
```

---

## Task 1: Модуль парсинга доменов и репозиторий сайтов

**Files:**
- Create: `src/agent/sites.py`
- Test: `tests/test_domains.py`

- [ ] **Step 1: Написать падающие тесты**

`tests/test_domains.py`:

```python
from src.agent.sites import Sites, extract_domains, normalize_domain, parse_sites_message


def test_normalize_domain_strips_scheme_path_and_www():
    assert normalize_domain("https://hh.ru/search/vacancy") == "hh.ru"
    assert normalize_domain("www.habr.com") == "habr.com"
    assert normalize_domain("HH.RU") == "hh.ru"


def test_normalize_domain_rejects_junk():
    assert normalize_domain("") is None
    assert normalize_domain("не сайт") is None


def test_extract_domains_dedupes_in_order():
    assert extract_domains("ищи на hh.ru и hh.ru, и на habr.com") == ["hh.ru", "habr.com"]


class _FakeGateway:
    def __init__(self, domains=None):
        self._domains = domains
        self.text_model = object()

    async def invoke_structured(self, model, messages, schema):
        if schema is Sites and self._domains is not None:
            return Sites(domains=self._domains)
        raise ValueError("unexpected schema")


async def test_parse_sites_uses_llm_first():
    gateway = _FakeGateway(domains=["rabota.ru"])
    assert await parse_sites_message(gateway, "ищи на rabota.ru") == ["rabota.ru"]


async def test_parse_sites_falls_back_to_regex():
    gateway = _FakeGateway(domains=None)
    assert await parse_sites_message(gateway, "ищи на hh.ru") == ["hh.ru"]
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_domains.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.sites'`

- [ ] **Step 3: Реализовать `src/agent/sites.py`**

```python
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class Sites(BaseModel):
    """Список доменов, распознанных из сообщения пользователя."""

    domains: list[str] = Field(default_factory=list)


_DOMAIN_RE = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:[a-z]{2,})", re.IGNORECASE
)


def normalize_domain(raw: str) -> str | None:
    """Нормализовать домен: убрать схему/путь/порт/www, привести к нижнему регистру."""
    value = raw.strip().lower()
    if not value:
        return None
    if "://" in value or value.startswith("www."):
        parsed = urlparse(value if "://" in value else f"//{value}")
        host = parsed.hostname
    else:
        host = value.split("/")[0].split(":")[0]
    if not host:
        return None
    host = host.removeprefix("www.")
    if "." not in host:
        return None
    return host


def extract_domains(text: str) -> list[str]:
    """Извлечь уникальные домены из текста (regex-фолбэк)."""
    result: list[str] = []
    for match in _DOMAIN_RE.findall(text):
        domain = normalize_domain(match)
        if domain and domain not in result:
            result.append(domain)
    return result


async def parse_sites_message(gateway, text: str) -> list[str]:
    """Распарсить сообщение в список доменов: сначала LLM, затем regex-фолбэк."""
    try:
        sites = await gateway.invoke_structured(
            gateway.text_model,
            [
                (
                    "system",
                    "Извлеки домены сайтов по поиску работы из сообщения. "
                    "Верни только hostname (без схемы, пути и www).",
                ),
                ("human", text),
            ],
            Sites,
        )
        domains = [d for d in (normalize_domain(x) for x in sites.domains) if d]
        if domains:
            return list(dict.fromkeys(domains))
    except Exception:
        pass
    return extract_domains(text)


class SitesRepository:
    """Доступ к сохранённым сайтам пользователя."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def get_domains(self, user_id: int) -> list[str]:
        from sqlalchemy import select

        from src.database.models import UserSite

        async with self._session_factory() as session:
            result = await session.execute(
                select(UserSite.domain)
                .where(UserSite.user_id == user_id)
                .order_by(UserSite.id)
            )
            return list(result.scalars().all())

    async def add_domains(self, user_id: int, domains: list[str]) -> None:
        from src.database.models import UserSite

        async with self._session_factory() as session:
            for domain in domains:
                session.add(UserSite(user_id=user_id, domain=domain))
            await session.commit()
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_domains.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Закоммитить**

```bash
git add src/agent/sites.py tests/test_domains.py
git commit -m "feat: domain parsing and sites repository"
```

---

## Task 2: Модель UserSite и миграция

**Files:**
- Modify: `src/database/models.py`
- Create: `migrations/versions/9a2b3c4d5e6f_add_user_sites.py`

- [ ] **Step 1: Добавить модель `UserSite`**

В `src/database/models.py` заменить импорт:

```python
from sqlalchemy import DateTime, ForeignKey, String, Text, func
```

на:

```python
from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
```

В конец файла добавить модель (после `User`, чтобы сохранить читаемый порядок):

```python
class UserSite(Base):
    """Сайт по поиску работы, добавленный пользователем."""

    __tablename__ = "user_sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    domain: Mapped[str] = mapped_column(String(255))
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("user_id", "domain", name="uq_user_sites_user_domain"),
    )
```

- [ ] **Step 2: Написать миграцию**

Создать `migrations/versions/9a2b3c4d5e6f_add_user_sites.py`:

```python
"""add user_sites

Revision ID: 9a2b3c4d5e6f
Revises: ea5108fb8c54
Create Date: 2026-08-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "9a2b3c4d5e6f"
down_revision: Union[str, Sequence[str], None] = "ea5108fb8c54"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_sites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "domain", name="uq_user_sites_user_domain"),
    )
    op.create_index(
        op.f("ix_user_sites_user_id"), "user_sites", ["user_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_user_sites_user_id"), table_name="user_sites")
    op.drop_table("user_sites")
```

- [ ] **Step 3: Применить миграцию**

Run (нужен запущенный postgres):

```bash
docker compose up -d postgres
uv run alembic upgrade head
```

Expected: выход без ошибок, таблица `user_sites` создана.

- [ ] **Step 4: Проверить, что модель импортируется**

Run: `uv run python -c "from src.database.models import UserSite; print(UserSite.__tablename__)"`
Expected: `user_sites`

- [ ] **Step 5: Закоммитить**

```bash
git add src/database/models.py migrations/versions/9a2b3c4d5e6f_add_user_sites.py
git commit -m "feat: add user_sites table and migration"
```

---

## Task 3: Whitelist доменов + scroll/back в browser-сервисе

**Files:**
- Modify: `src/browser/server.py`
- Modify: `src/browser/executor.py`
- Test: `tests/test_browser.py`

- [ ] **Step 1: Написать падающие тесты**

Добавить в конец `tests/test_browser.py`:

```python
async def test_navigate_with_allowed_domains(mock_site, browser_client):
    r = await browser_client.post(
        "/navigate",
        json={
            "user_id": 1,
            "url": f"{mock_site}/index.html",
            "allowed_domains": ["127.0.0.1"],
        },
    )
    assert r.status_code == 200


async def test_navigate_rejects_domain_not_in_allowed_list(browser_client):
    r = await browser_client.post(
        "/navigate",
        json={"user_id": 1, "url": "https://evil.com", "allowed_domains": ["example.com"]},
    )
    assert r.status_code == 403


async def test_scroll_and_back(mock_site, browser_client):
    await browser_client.post(
        "/navigate",
        json={"user_id": 1, "url": f"{mock_site}/index.html"},
    )
    r = await browser_client.post("/scroll", json={"user_id": 1, "delta": 300})
    assert r.status_code == 200
    r = await browser_client.post("/back", json={"user_id": 1})
    assert r.status_code == 200
```

- [ ] **Step 2: Запустить, убедиться что падают**

Run: `uv run pytest tests/test_browser.py::test_navigate_with_allowed_domains tests/test_browser.py::test_scroll_and_back -v`
Expected: FAIL (422 или 404 — эндпоинты/поля ещё не существуют)

- [ ] **Step 3: Изменить `src/browser/server.py`**

Заменить `NavigateRequest` и добавить `ScrollRequest` (после `NavigateRequest`):

```python
class NavigateRequest(BaseModel):
    """Тело запроса навигации на URL."""

    user_id: int
    url: str
    allowed_domains: list[str] | None = None


class ScrollRequest(BaseModel):
    """Тело запроса прокрутки страницы."""

    user_id: int
    delta: int = 800
```

Заменить `_is_allowed_url`:

```python
def _is_allowed_url(url: str, allowed_domains: list[str] | None = None) -> bool:
    """Проверить URL: схема http/https и домен в whitelist (или его поддомен).

    :param url: целевой URL
    :param allowed_domains: список разрешённых доменов; None — статический whitelist
    :return: True, если навигация разрешена
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname or ""
    domains = allowed_domains if allowed_domains is not None else ALLOWED_DOMAINS
    return any(hostname == d or hostname.endswith("." + d) for d in domains)
```

Добавить методы `scroll` и `back` в `BrowserManager` (после `screenshot`):

```python
    async def scroll(self, user_id: int, delta: int = 800) -> dict:
        """Прокрутить страницу на delta пикселей вниз."""
        await self._throttle(user_id)
        page = await self.get_page(user_id)
        await page.mouse.wheel(0, delta)
        return {"ok": True}

    async def back(self, user_id: int) -> dict:
        """Вернуться на предыдущую страницу."""
        await self._throttle(user_id)
        page = await self.get_page(user_id)
        await page.go_back()
        return {"ok": True}
```

Заменить эндпоинт `/navigate`:

```python
    @app.post("/navigate")
    async def navigate(req: NavigateRequest) -> dict:
        if not _is_allowed_url(req.url, req.allowed_domains):
            raise HTTPException(status_code=403, detail=f"Домен не разрешён: {req.url}")
        return await manager.navigate(req.user_id, req.url)
```

Добавить эндпоинты `/scroll` и `/back` (после `/screenshot`):

```python
    @app.post("/scroll")
    async def scroll(req: ScrollRequest) -> dict:
        return await manager.scroll(req.user_id, req.delta)

    @app.post("/back")
    async def back(req: UserRequest) -> dict:
        return await manager.back(req.user_id)
```

- [ ] **Step 4: Изменить `src/browser/executor.py`**

Заменить `navigate`:

```python
    async def navigate(
        self, user_id: int, url: str, allowed_domains: list[str] | None = None
    ) -> dict:
        """Перейти на URL в браузере пользователя.

        :param user_id: идентификатор пользователя
        :param url: целевой URL
        :param allowed_domains: список разрешённых доменов для проверки на сервере
        :return: словарь с итоговым url и title
        """
        payload: dict = {"user_id": user_id, "url": url}
        if allowed_domains is not None:
            payload["allowed_domains"] = allowed_domains
        r = await self._client.post("/navigate", json=payload)
        r.raise_for_status()
        return r.json()
```

Добавить методы `scroll` и `back` (после `screenshot`):

```python
    async def scroll(self, user_id: int, delta: int = 800) -> dict:
        """Прокрутить страницу на delta пикселей вниз."""
        r = await self._client.post(
            "/scroll", json={"user_id": user_id, "delta": delta}
        )
        r.raise_for_status()
        return r.json()

    async def back(self, user_id: int) -> dict:
        """Вернуться на предыдущую страницу."""
        r = await self._client.post("/back", json={"user_id": user_id})
        r.raise_for_status()
        return r.json()
```

- [ ] **Step 5: Запустить тесты browser-модуля**

Run: `uv run pytest tests/test_browser.py -v`
Expected: PASS (все тесты, включая новые)

- [ ] **Step 6: Закоммитить**

```bash
git add src/browser/server.py src/browser/executor.py tests/test_browser.py
git commit -m "feat: per-user domain whitelist, scroll and back actions"
```

---

## Task 4: Браузерный цикл с VLM (`BrowserLoop`)

**Files:**
- Create: `src/agent/browser_loop.py`
- Test: `tests/test_browser_loop.py`

- [ ] **Step 1: Написать падающие тесты**

`tests/test_browser_loop.py`:

```python
from src.agent.browser_loop import BrowserAction, BrowserLoop, SearchOutcome


class _FakeModel:
    pass


class FakeLoopGateway:
    def __init__(self, actions):
        self._actions = list(actions)
        self.vision_model = _FakeModel()

    async def invoke_structured(self, model, messages, schema):
        assert schema is BrowserAction
        if not self._actions:
            return BrowserAction(tool="done", args={})
        return self._actions.pop(0)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def extract(self, user_id):
        return {"url": "https://example.com", "title": "t", "text": "", "elements": []}

    async def screenshot(self, user_id):
        return b"\x89PNG\r\n\x1a\nfake"

    async def navigate(self, user_id, url, allowed_domains=None):
        self.calls.append(("navigate", url, allowed_domains))
        return {}

    async def click(self, user_id, selector):
        self.calls.append(("click", selector))
        return {}

    async def type_text(self, user_id, selector, text):
        self.calls.append(("type", selector, text))
        return {}

    async def scroll(self, user_id, delta=800):
        self.calls.append(("scroll", delta))
        return {}

    async def back(self, user_id):
        self.calls.append(("back",))
        return {}


async def test_loop_executes_actions_until_done():
    actions = [
        BrowserAction(tool="click", args={"selector": "#search"}),
        BrowserAction(tool="type", args={"selector": "#search", "text": "python"}),
        BrowserAction(
            tool="done",
            args={"candidates": [{"title": "Dev", "url": "https://example.com/1"}]},
        ),
    ]
    gateway = FakeLoopGateway(actions)
    executor = FakeExecutor()
    loop = BrowserLoop(executor, gateway, max_steps=10)
    outcome = await loop.run(
        1, "goal", SearchOutcome, allowed_domains=["example.com"],
        start_url="https://example.com",
    )
    assert len(outcome.candidates) == 1
    assert executor.calls[0] == ("navigate", "https://example.com", ["example.com"])
    assert ("click", "#search") in executor.calls
    assert ("type", "#search", "python") in executor.calls


async def test_loop_returns_empty_on_step_limit():
    actions = [BrowserAction(tool="click", args={"selector": "#x"})] * 100
    gateway = FakeLoopGateway(actions)
    executor = FakeExecutor()
    loop = BrowserLoop(executor, gateway, max_steps=2)
    outcome = await loop.run(1, "goal", SearchOutcome, allowed_domains=[])
    assert outcome.candidates == []
```

- [ ] **Step 2: Запустить, убедиться что падают**

Run: `uv run pytest tests/test_browser_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.browser_loop'`

- [ ] **Step 3: Реализовать `src/agent/browser_loop.py`**

```python
import base64
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class BrowserAction(BaseModel):
    """Одно действие браузера, выбранное VLM."""

    tool: Literal["navigate", "click", "type", "scroll", "back", "done"]
    args: dict = Field(default_factory=dict)


class SearchOutcome(BaseModel):
    """Результат поиска: найденные вакансии."""

    candidates: list[dict] = Field(default_factory=list)


class ApplyOutcome(BaseModel):
    """Результат отклика."""

    applied: bool = False
    detail: str = ""


_SYSTEM_PROMPT = (
    "Ты управляешь браузером для поиска работы. На каждом шаге получаешь "
    "скриншот страницы и список интерактивных элементов с CSS-селекторами. "
    "Верни одно действие: tool из [navigate, click, type, scroll, back, done]. "
    "Для click/type используй selector ТОЛЬКО из списка элементов. "
    "Когда цель достигнута, верни tool=done и результат в args."
)


class BrowserLoop:
    """Цикл «скриншот → VLM → действие», пока цель не достигнута."""

    def __init__(self, executor, gateway, max_steps: int | None = None) -> None:
        self._executor = executor
        self._gateway = gateway
        self._max_steps = max_steps if max_steps is not None else settings.BROWSER_MAX_STEPS

    async def run(
        self,
        user_id: int,
        goal: str,
        result_schema: type,
        allowed_domains: list[str],
        start_url: str | None = None,
    ):
        if start_url:
            await self._executor.navigate(
                user_id, start_url, allowed_domains=allowed_domains
            )
        for _ in range(self._max_steps):
            page = await self._executor.extract(user_id)
            shot = await self._executor.screenshot(user_id)
            action = await self._decide(goal, page, shot)
            if action.tool == "done":
                return result_schema.model_validate(action.args)
            await self._execute(user_id, action, allowed_domains)
        return result_schema()

    async def _decide(self, goal: str, page: dict, shot: bytes) -> BrowserAction:
        image = base64.b64encode(shot).decode()
        elements = page.get("elements", [])
        text = (page.get("text") or "")[:4000]
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Цель: {goal}\n\nТекст страницы:\n{text}\n\n"
                            f"Элементы (селекторы):\n{elements}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ]
            ),
        ]
        try:
            return await self._gateway.invoke_structured(
                self._gateway.vision_model, messages, BrowserAction
            )
        except Exception as exc:
            logger.warning("VLM decision failed, retrying: %s", exc)
            return await self._gateway.invoke_structured(
                self._gateway.vision_model, messages, BrowserAction
            )

    async def _execute(
        self, user_id: int, action: BrowserAction, allowed_domains: list[str]
    ) -> None:
        tool = action.tool
        args = action.args or {}
        try:
            if tool == "navigate":
                await self._executor.navigate(
                    user_id, args["url"], allowed_domains=allowed_domains
                )
            elif tool == "click":
                await self._executor.click(user_id, args["selector"])
            elif tool == "type":
                await self._executor.type_text(
                    user_id, args["selector"], args["text"]
                )
            elif tool == "scroll":
                await self._executor.scroll(user_id, args.get("delta", 800))
            elif tool == "back":
                await self._executor.back(user_id)
        except Exception as exc:
            logger.warning("Browser action %s failed: %s", tool, exc)
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_browser_loop.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Закоммитить**

```bash
git add src/agent/browser_loop.py tests/test_browser_loop.py
git commit -m "feat: VLM-driven browser loop"
```

---

## Task 5: UniversalSearcher и UniversalApplier

**Files:**
- Create: `src/agent/searcher.py`
- Test: `tests/test_searcher.py`

- [ ] **Step 1: Написать падающие тесты**

`tests/test_searcher.py`:

```python
from src.agent.browser_loop import ApplyOutcome, SearchOutcome
from src.agent.searcher import UniversalApplier, UniversalSearcher


class FakeSites:
    def __init__(self, domains):
        self._domains = list(domains)

    async def get_domains(self, user_id):
        return self._domains

    async def add_domains(self, user_id, domains):
        pass


class FakeLoop:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.runs = []

    async def run(self, user_id, goal, result_schema, allowed_domains, start_url=None):
        self.runs.append((goal, result_schema, allowed_domains, start_url))
        return self._outcomes.pop(0)


async def test_universal_searcher_collects_from_all_sites():
    outcomes = [
        SearchOutcome(candidates=[{"title": "A", "url": "https://a.com/1"}]),
        SearchOutcome(candidates=[]),
    ]
    searcher = UniversalSearcher(
        executor=None,
        gateway=None,
        sites=FakeSites(["a.com", "b.com"]),
        loop=FakeLoop(outcomes),
    )
    result = await searcher(1, "python")
    assert len(result) == 1
    assert result[0]["site"] == "a.com"
    assert len(searcher._loop.runs) == 2
    assert searcher._loop.runs[0][3] == "https://a.com"


async def test_searcher_prefers_desired_role():
    async def profile(user_id):
        return {"desired_role": "data engineer"}

    searcher = UniversalSearcher(
        executor=None,
        gateway=None,
        sites=FakeSites(["a.com"]),
        profile_provider=profile,
        loop=FakeLoop([SearchOutcome()]),
    )
    await searcher(1, "посмотри что нового")
    goal = searcher._loop.runs[0][0]
    assert "data engineer" in goal


async def test_universal_applier_runs_apply_loop():
    applier = UniversalApplier(
        executor=None,
        gateway=None,
        sites=FakeSites(["a.com"]),
        loop=FakeLoop([ApplyOutcome(applied=True)]),
    )
    result = await applier(
        1, {"job": {"url": "https://a.com/job/1"}, "cover_letter": "Привет"}
    )
    assert result.applied is True
    goal, schema, allowed, start = applier._loop.runs[0]
    assert "https://a.com/job/1" in goal
    assert start == "https://a.com/job/1"
    assert allowed == ["a.com"]
```

- [ ] **Step 2: Запустить, убедиться что падают**

Run: `uv run pytest tests/test_searcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.searcher'`

- [ ] **Step 3: Реализовать `src/agent/searcher.py`**

```python
from src.agent.browser_loop import ApplyOutcome, BrowserLoop, SearchOutcome


class UniversalSearcher:
    """Поиск вакансий на всех сайтах пользователя через VLM-цикл."""

    def __init__(
        self, executor, gateway, sites, profile_provider=None, loop=None
    ) -> None:
        self._loop = loop or BrowserLoop(executor, gateway)
        self._sites = sites
        self._profile_provider = profile_provider

    async def __call__(self, user_id: int, query: str = "") -> list[dict]:
        domains = await self._sites.get_domains(user_id)
        search_query = await self._resolve_query(user_id, query)
        candidates: list[dict] = []
        for domain in domains:
            goal = (
                f"Найди на сайте {domain} поле поиска вакансий, введи запрос "
                f"«{search_query}», получи список вакансий со страницы результатов "
                "и собери для каждой: title, url, короткое описание."
            )
            outcome: SearchOutcome = await self._loop.run(
                user_id, goal, SearchOutcome, allowed_domains=domains,
                start_url=f"https://{domain}",
            )
            for candidate in outcome.candidates:
                candidate.setdefault("site", domain)
            candidates.extend(outcome.candidates)
        return candidates

    async def _resolve_query(self, user_id: int, query: str) -> str:
        if self._profile_provider is not None:
            profile = await self._profile_provider(user_id)
            role = (profile or {}).get("desired_role")
            if role:
                return role
        return query


class UniversalApplier:
    """Отклик на вакансию через VLM-цикл."""

    def __init__(self, executor, gateway, sites, loop=None) -> None:
        self._loop = loop or BrowserLoop(executor, gateway)
        self._sites = sites

    async def __call__(self, user_id: int, decision: dict) -> ApplyOutcome:
        url = decision.get("job", {}).get("url", "")
        cover = decision.get("cover_letter", "")
        domains = await self._sites.get_domains(user_id)
        goal = (
            f"Открой вакансию {url}, найди кнопку отклика (Откликнуться/Apply), "
            f"нажми её, впиши сопроводительное письмо: «{cover}», отправь и "
            "подтверди успех."
        )
        return await self._loop.run(
            user_id, goal, ApplyOutcome, allowed_domains=domains, start_url=url
        )
```

- [ ] **Step 4: Запустить тесты, убедиться что проходят**

Run: `uv run pytest tests/test_searcher.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Закоммитить**

```bash
git add src/agent/searcher.py tests/test_searcher.py
git commit -m "feat: universal searcher and applier"
```

---

## Task 6: Onboarding + composition root + удаление hh-специфики

**Files:**
- Modify: `src/tg/service.py`
- Modify: `src/tg/bot.py`
- Modify: `src/browser/server.py` (удалить `/submit`, инлайнить chat-селекторы)
- Modify: `src/browser/executor.py` (удалить `submit_application`)
- Delete: `src/browser/searcher.py`, `src/browser/adapters.py`
- Test: `tests/test_tg.py` (обновить + onboarding-тесты)
- Test: `tests/test_browser.py` (удалить `test_submit_application`)

- [ ] **Step 1: Обновить тесты TgService**

Заменить в `tests/test_tg.py` блок после `_fake_session_factory`:

```python
class FakeSitesRepo:
    """Поддельный репозиторий сайтов: заданный список + фиксация добавлений."""

    def __init__(self, domains):
        self._domains = list(domains)
        self.added = []

    async def get_domains(self, user_id):
        return self._domains

    async def add_domains(self, user_id, domains):
        self.added.extend(domains)
        self._domains.extend(domains)


def _build_service(
    gateway: FakeGateway,
    searcher: FakeSearcher | None = None,
    sites_repo: FakeSitesRepo | None = None,
) -> TgService:
    """Собрать TgService с фейками и in-memory чекпоинтером."""
    searcher = searcher or FakeSearcher([])
    sites_repo = sites_repo if sites_repo is not None else FakeSitesRepo(["hh.ru"])
    graph = build_graph(gateway, searcher, checkpointer=MemorySaver())
    return TgService(
        graph, _fake_session_factory(), gateway=gateway, sites_repo=sites_repo
    )
```

Добавить в конец `tests/test_tg.py` onboarding-тесты:

```python
async def test_onboarding_asks_for_sites_when_none():
    """Нет сайтов: бот спрашивает, на каких сайтах искать, и не запускает граф."""
    service = _build_service(FakeGateway(intent="chat"), sites_repo=FakeSitesRepo([]))
    reply = await service.handle_message("123", "привет")
    assert "На каких сайтах" in reply


async def test_onboarding_saves_sites_from_message():
    """Сообщение с доменами при отсутствии сайтов сохраняет их и подтверждает."""
    sites_repo = FakeSitesRepo([])
    service = _build_service(FakeGateway(intent="chat"), sites_repo=sites_repo)
    reply = await service.handle_message("123", "ищи на hh.ru и habr.com")
    assert "hh.ru" in reply
    assert "habr.com" in reply
    assert "hh.ru" in sites_repo.added
    assert "habr.com" in sites_repo.added
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_tg.py -v`
Expected: FAIL — `TgService.__init__` ещё не принимает `gateway`/`sites_repo`, onboarding-тесты падают на `AttributeError`/assert.

- [ ] **Step 3: Обновить `src/tg/service.py`**

Добавить импорт:

```python
from src.agent.sites import SitesRepository, parse_sites_message
```

Заменить `__init__`:

```python
    def __init__(self, graph, session_factory, gateway=None, sites_repo=None) -> None:
        self._graph = graph
        self._session_factory = session_factory
        self._gateway = gateway
        self._sites = sites_repo or SitesRepository(session_factory)
```

Заменить начало `handle_message` (до `config = ...`):

```python
    async def handle_message(self, telegram_id: str, text: str) -> str:
        user_id = await self._get_or_create_user(telegram_id)

        # Onboarding: без сайтов просим назвать площадки и не запускаем граф
        domains = await self._sites.get_domains(user_id)
        if not domains:
            parsed = await parse_sites_message(self._gateway, text)
            if parsed:
                await self._sites.add_domains(user_id, parsed)
                return f"Запомнил. Буду искать на: {', '.join(parsed)}."
            return "На каких сайтах искать работу? Назови сайты через запятую."

        config = {"configurable": {"thread_id": f"tg-{telegram_id}"}}
```

(Остальная часть `handle_message` — HITL-возобновление и `_build_reply` — не меняется.)

- [ ] **Step 4: Обновить `src/tg/bot.py` (composition root)**

Заменить импорты:

```python
from src.agent.router import IntentRouter
from src.agent.sites import SitesRepository
from src.agent.searcher import UniversalApplier, UniversalSearcher
from src.browser.executor import BrowserExecutor
from src.core.audit import log_action
from src.database.db_settings import SessionLocal
from src.llm.gateway import LLMGateway
from src.stats.aggregate import get_user_stats
from src.tg.service import TgService
```

(Удалить `from src.browser.searcher import HhSearcher`.)

Заменить `build_tg_service`:

```python
def build_tg_service() -> TgService:
    """Собрать TgService со всеми зависимостями (композиционный корень)."""
    from langgraph.checkpoint.memory import MemorySaver

    gateway = LLMGateway()
    browser = BrowserExecutor(settings.BROWSER_EXECUTOR_URL)
    sites_repo = SitesRepository(SessionLocal)
    searcher = UniversalSearcher(
        browser, gateway, sites_repo, profile_provider=_get_profile
    )
    applier = UniversalApplier(browser, gateway, sites_repo)

    async def _apply(user_id: int, decision: dict) -> None:
        outcome = await applier(user_id, decision)
        log_action(
            user_id,
            "apply",
            "executed" if outcome.applied else "failed",
            {"job": decision.get("job", {})},
        )

    graph = build_graph(
        gateway,
        searcher,
        router=IntentRouter(gateway),
        checkpointer=MemorySaver(),
        get_stats=_get_stats,
        applier=_apply,
        profile_provider=_get_profile,
    )
    return TgService(graph, SessionLocal, gateway=gateway, sites_repo=sites_repo)
```

- [ ] **Step 5: Удалить hh-специфику из browser-сервиса**

В `src/browser/server.py`:

1. Удалить `SubmitRequest` (класс после `MessageRequest`).
2. Удалить метод `submit_application` из `BrowserManager`.
3. Удалить эндпоинт `@app.post("/submit")`.
4. Убрать импорт `from src.browser.adapters import HhAdapter` и заменить его константами (в начале файла, после `_EXTRACT_JS`):

```python
_CHAT_INPUT_SELECTORS = (
    "textarea[data-qa='messenger-input'], textarea[data-qa='chat_input']"
)
_CHAT_SEND_SELECTORS = (
    "button[data-qa='messenger-send'], button[data-qa='chat_send_button']"
)
```

5. В `send_message` заменить `HhAdapter.CHAT_INPUT` → `_CHAT_INPUT_SELECTORS`,
   `HhAdapter.CHAT_SEND` → `_CHAT_SEND_SELECTORS`.

В `src/browser/executor.py`: удалить метод `submit_application` целиком.

Удалить файлы `src/browser/searcher.py` и `src/browser/adapters.py`:

```bash
rm src/browser/searcher.py src/browser/adapters.py
```

- [ ] **Step 6: Удалить `test_submit_application`**

В `tests/test_browser.py` удалить тест `test_submit_application` (целиком).

- [ ] **Step 7: Запустить тесты**

Run: `uv run pytest tests/test_tg.py tests/test_browser.py -v`
Expected: PASS

- [ ] **Step 8: Закоммитить**

```bash
git add src/tg/service.py src/tg/bot.py src/browser/server.py src/browser/executor.py tests/test_tg.py tests/test_browser.py
git rm src/browser/searcher.py src/browser/adapters.py
git commit -m "feat: onboarding + universal searcher/applier wiring, drop hh-specific code"
```

---

## Task 7: Конфиг, .env.example, README

**Files:**
- Modify: `config.py`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Добавить `BROWSER_MAX_STEPS` в `config.py`**

После блока `# Browser executor` добавить строку:

```python
    BROWSER_MAX_STEPS: int = 15  # лимит шагов VLM-цикла браузера
```

- [ ] **Step 2: Добавить в `.env.example`**

В конец файла добавить:

```bash
BROWSER_MAX_STEPS=15
```

- [ ] **Step 3: Обновить README (vision-модель + мульти-сайт)**

В разделе «Какие модели установить» заменить строку про vision-модель и добавить примечание про мульти-сайтовый VLM:

```markdown
ollama pull qwen3-vl:4b       # vision-модель для разбора страниц (или llama3.2-vision)
```

После пункта «3. Какие модели установить» добавить абзац:

```markdown
Vision-модель (`LLM_VISION_MODEL`) теперь реально используется агентом: она
«смотрит» на страницы и управляет браузером при поиске и отклике на любых сайтах.
Для Ollama подойдёт `qwen3-vl:4b` (или `llama3.2-vision`), для облака — `gpt-4o`.
При первом общении бот спросит, на каких сайтах искать, и запомнит список для
каждого пользователя.
```

- [ ] **Step 4: Закоммитить**

```bash
git add config.py .env.example README.md
git commit -m "docs: multi-site VLM config and README updates"
```

---

## Task 8: Финальная проверка

- [ ] **Step 1: Полный прогон тестов**

Run: `uv run pytest -q`
Expected: PASS (тесты с Docker-зависимостью пропускаются, если Docker недоступен).

- [ ] **Step 2: Линт**

Run: `uv run black --check src tests config.py main.py`
Expected: без ошибок (при необходимости `uv run black src tests config.py main.py`).

- [ ] **Step 3: Импорт-дым-тест**

Run: `uv run python -c "from src.tg.bot import build_tg_service; print('ok')"`
Expected: `ok`

---

## Self-Review Notes

- **Spec coverage:** onboarding (Task 6), whitelist (Task 3), VLM-цикл search/apply
  (Task 4/5), удаление hh-специфики (Task 6), конфиг/провайдеры (Task 7), тесты
  (Tasks 1–8) — все пункты спецификации покрыты.
- **Упрощение относительно спецификации:** onboarding определяется отсутствием
  сайтов (`user_sites` пуст), а не флагом `User.status`. Это убирает мутацию
  статуса и не ломает существующие записи пользователей; поведение эквивалентно.
- **Отклонение от спецификации:** chat-селекторы из `HhAdapter` инлайнятся в
  `server.py`, т.к. `send_message` (вне scope) всё ещё их использует.
