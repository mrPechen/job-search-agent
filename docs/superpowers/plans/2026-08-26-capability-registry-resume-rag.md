# Capability Registry + Resume Upload + RAG Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent a capability registry (routes text to handlers or says "не умею"), accept a PDF/DOCX resume in Telegram, and use RAG (chunked resume) during vacancy scoring.

**Architecture:** A dispatcher in `TgService` routes text via `CapabilityRouter` (LLM + keyword fallback) to a handler; the LangGraph graph shrinks to the search pipeline (`search → match → decision → apply → report` + HITL). Resume upload parses the file into a structured `Profile` plus chunked `documents` embeddings; scoring injects relevant resume chunks into the LLM prompt.

**Tech Stack:** Python 3.12, LangGraph, LangChain (ollama/openai), aiogram, SQLAlchemy async + pgvector, pytest-asyncio, testcontainers.

**Spec:** `docs/superpowers/specs/2026-08-26-capability-registry-resume-rag-design.md`

---

## Current baseline

Before starting, the test suite has one known failure:

```bash
uv run pytest -q
# 1 failed (tests/test_rag.py::test_store_and_retrieve_relevant), 64 passed
```

The failure is the dimension mismatch: ORM `Document.embedding` reads
`Vector(settings.EMBEDDING_DIM)` = 768 (from `.env`), but `test_store_and_retrieve_relevant`
uses 1536-dim vectors. This is fixed in Task 1.

---

## Task 1: Fix embedding dimension in `tests/test_rag.py`

**Files:**
- Modify: `tests/test_rag.py:84-96`

The test must use 768-dim vectors to match `EMBEDDING_DIM=768` (ollama
`nomic-embed-text`).

- [ ] **Step 1: Edit the dimension literals**

In `tests/test_rag.py`, change lines 84-96 to use 768:

```python
    # Чанк A близок к запросу, чанк B — противоположен
    emb_a = [1.0] * 768
    emb_b = [-1.0] * 768
    async with session_factory() as session:
        count = await store_chunks(
            session, 1, ["Python developer", "Sales manager"], [emb_a, emb_b]
        )
        await session.commit()
        assert count == 2

    async with session_factory() as session:
        result = await retrieve_relevant(session, 1, [1.0] * 768, top_k=2)
        assert len(result) == 2
        assert result[0].chunk_text == "Python developer"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_rag.py -q`
Expected: `6 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag.py
git commit -m "test: use 768-dim vectors to match EMBEDDING_DIM"
```

---

## Task 2: Add `embedder` to `LLMGateway`

**Files:**
- Modify: `src/llm/gateway.py`
- Test: `tests/test_llm.py` (extend)

- [ ] **Step 1: Add a failing test for the embedder**

Read `tests/test_llm.py` first (currently 9 lines testing `build_text_model`/`build_vision_model`).
Add this test:

```python
def test_gateway_exposes_embedder():
    """gateway.embedder собирается через build_embeddings и не падает."""
    from src.llm.gateway import LLMGateway

    gateway = LLMGateway()
    embedder = gateway.embedder
    assert embedder is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -q`
Expected: FAIL with `AttributeError: 'LLMGateway' object has no attribute 'embedder'`

- [ ] **Step 3: Implement `embedder` in `src/llm/gateway.py`**

Add after the `vision_model` cached_property (around line 38):

```python
    @cached_property
    def embedder(self):
        """Модель эмбеддингов для RAG (провайдер совпадает с LLM_PROVIDER)."""
        from src.rag.ingest import build_embeddings

        return build_embeddings(
            settings.LLM_PROVIDER,
            settings.EMBEDDING_MODEL,
            settings.OPENAI_API_KEY,
            settings.OLLAMA_BASE_URL,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -q`
Expected: PASS (building `OllamaEmbeddings`/`OpenAIEmbeddings` makes no network call)

- [ ] **Step 5: Commit**

```bash
git add src/llm/gateway.py tests/test_llm.py
git commit -m "feat: expose embedder on LLMGateway for RAG"
```

---

## Task 3: Add `ProfileRepository` (`src/rag/profile_repo.py`)

**Files:**
- Create: `src/rag/profile_repo.py`
- Test: `tests/test_rag.py` (extend)

- [ ] **Step 1: Add a failing test**

Append to `tests/test_rag.py` (reuse the module-scoped `pg_url` fixture and imports):

```python
async def test_profile_repository_upsert(pg_url):
    """Сохранение профиля: вторая запись обновляет существующую, не создавая дубль."""
    from sqlalchemy import select

    from src.database.models import Profile
    from src.rag.profile_repo import ProfileRepository
    from src.rag.schemas import ProfileData

    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        session.add(User(id=1, telegram_id="t1"))
        await session.commit()

    repo = ProfileRepository(session_factory)
    await repo.save(1, ProfileData(skills=["python"], desired_role="dev"))
    await repo.save(1, ProfileData(skills=["python", "sql"], desired_role="dev"))

    async with session_factory() as session:
        result = await session.execute(select(Profile).where(Profile.user_id == 1))
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].skills == ["python", "sql"]

    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag.py::test_profile_repository_upsert -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.profile_repo'`

- [ ] **Step 3: Create `src/rag/profile_repo.py`**

```python
from src.rag.schemas import ProfileData


class ProfileRepository:
    """Доступ к профилю соискателя: upsert структурированных данных CV."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def save(self, user_id: int, data: ProfileData) -> None:
        """Сохранить профиль: обновить существующий или создать новый."""
        from sqlalchemy import select

        from src.database.models import Profile

        async with self._session_factory() as session:
            result = await session.execute(
                select(Profile).where(Profile.user_id == user_id)
            )
            profile = result.scalars().first()
            if profile is None:
                profile = Profile(user_id=user_id)
                session.add(profile)
            profile.skills = data.skills
            profile.experience = data.experience
            profile.desired_role = data.desired_role
            profile.desired_salary = data.desired_salary
            profile.desired_location = data.desired_location
            await session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag.py::test_profile_repository_upsert -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/profile_repo.py tests/test_rag.py
git commit -m "feat: add ProfileRepository for CV profile upsert"
```

---

## Task 4: Add `ResumeRetriever` and `ingest_resume_chunks` (`src/rag/retrieve.py`)

**Files:**
- Modify: `src/rag/retrieve.py`
- Test: `tests/test_rag.py` (extend)

- [ ] **Step 1: Add a failing test**

Append to `tests/test_rag.py`:

```python
async def test_resume_retriever_returns_chunk_texts(pg_url):
    """Ретривер эмбеддит запрос и возвращает тексты ближайших чанков."""
    from src.rag.retrieve import ResumeRetriever

    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        session.add(User(id=1, telegram_id="t1"))
        await session.commit()

    async with session_factory() as session:
        await store_chunks(
            session, 1, ["Python developer", "Sales manager"], [[1.0] * 768, [-1.0] * 768]
        )
        await session.commit()

    class FakeEmbedder:
        async def aembed_query(self, query: str) -> list[float]:
            return [1.0] * 768

    retriever = ResumeRetriever(session_factory, FakeEmbedder())
    texts = await retriever(1, "python backend")

    assert texts[0] == "Python developer"

    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag.py::test_resume_retriever_returns_chunk_texts -q`
Expected: FAIL with `ImportError: cannot import name 'ResumeRetriever'`

- [ ] **Step 3: Implement in `src/rag/retrieve.py`**

Add imports at the top of `src/rag/retrieve.py` (keep existing imports):

```python
from src.rag.ingest import chunk_by_sections, embed_chunks
```

Append at the bottom of the file:

```python
async def ingest_resume_chunks(
    session_factory, embedder, user_id: int, text: str
) -> int:
    """Чанковать резюме, эмбеддить и сохранить. Возвращает число чанков."""
    chunks = chunk_by_sections(text)
    if not chunks:
        return 0
    embeddings = await embed_chunks(embedder, chunks)
    async with session_factory() as session:
        count = await store_chunks(session, user_id, chunks, embeddings)
        await session.commit()
        return count


class ResumeRetriever:
    """Извлечение релевантных фрагментов резюме для скоринга вакансии."""

    def __init__(self, session_factory, embedder) -> None:
        self._session_factory = session_factory
        self._embedder = embedder

    async def __call__(self, user_id: int, query: str) -> list[str]:
        """Вернуть тексты top-k чанков, близких к запросу."""
        if not query:
            return []
        query_embedding = await self._embedder.aembed_query(query)
        async with self._session_factory() as session:
            docs = await retrieve_relevant(session, user_id, query_embedding, top_k=5)
        return [d.chunk_text for d in docs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag.py::test_resume_retriever_returns_chunk_texts -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieve.py tests/test_rag.py
git commit -m "feat: add ResumeRetriever and ingest_resume_chunks for RAG"
```

---

## Task 5: Create capabilities module (`src/agent/capabilities.py`)

**Files:**
- Create: `src/agent/capabilities.py`
- Create: `tests/test_capabilities.py`
- Delete: `tests/test_router.py` (replaced by `test_capabilities.py`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capabilities.py`:

```python
from src.agent.capabilities import Capability, CapabilityRouter


class _FakeModel:
    async def ainvoke(self, messages):
        return type("Msg", (), {"content": "ok"})()


class _FakeGateway:
    """Фейковый gateway: возвращает заданный ключ или бросает исключение."""

    def __init__(self, key: str | None = None, fail: bool = False) -> None:
        self.key = key
        self.fail = fail
        self.text_model = _FakeModel()

    async def invoke_structured(self, model, messages, schema):
        if self.fail:
            raise RuntimeError("llm down")
        from src.agent.capabilities import CapabilityChoice

        return CapabilityChoice(key=self.key)


_CAPS = [
    Capability(key="upload_resume", description="принять резюме"),
    Capability(key="search_job", description="искать работу"),
    Capability(key="stats", description="статистика"),
    Capability(key="chat", description="свободная переписка"),
]


async def test_classify_returns_key():
    router = CapabilityRouter(_FakeGateway(key="stats"), _CAPS)
    assert await router.classify("сколько откликов") == "stats"


async def test_classify_unknown_key_falls_to_none():
    router = CapabilityRouter(_FakeGateway(key="bogus"), _CAPS)
    assert await router.classify("сделай то") is None


async def test_classify_llm_none():
    router = CapabilityRouter(_FakeGateway(key=None), _CAPS)
    assert await router.classify("отправь на почту") is None


async def test_classify_keyword_fallback():
    router = CapabilityRouter(_FakeGateway(fail=True), _CAPS)
    assert await router.classify("куда прислать резюме") == "upload_resume"
    assert await router.classify("сколько откликов") == "stats"
    assert await router.classify("найди работу") == "search_job"
    assert await router.classify("привет") == "chat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_capabilities.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.capabilities'`

- [ ] **Step 3: Create `src/agent/capabilities.py`**

```python
import re

from pydantic import BaseModel


class Capability(BaseModel):
    """Одна способность агента: ключ + описание для LLM-промпта."""

    key: str
    description: str


class CapabilityChoice(BaseModel):
    """Результат выбора способности. key=None — ни одна не подходит."""

    key: str | None = None


# Ключевые слова для детерминированного fallback (по границе слова)
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "upload_resume": ("резюме", "резюм", "cv"),
    "stats": ("стат", "сколько", "итог", "результат"),
    "search_job": ("работ", "ваканс", "ищ", "поиск", "найти", "собесед"),
}


def _has_keyword(lower: str, words: tuple[str, ...]) -> bool:
    """Проверить вхождение любого ключевого слова по границе слова."""
    return any(re.search(rf"\b{re.escape(word)}", lower) for word in words)


class CapabilityRouter:
    """Сопоставление сообщения со способностью агента (LLM + keyword-fallback)."""

    def __init__(self, gateway, capabilities: list[Capability]) -> None:
        self._gateway = gateway
        self._capabilities = capabilities

    @property
    def capabilities(self) -> list[Capability]:
        """Список доступных способностей (для перечня «я умею»)."""
        return self._capabilities

    async def classify(self, message: str) -> str | None:
        """Вернуть ключ способности или None, если ни одна не подходит."""
        try:
            choice = await self._gateway.invoke_structured(
                self._gateway.text_model,
                [
                    ("system", self._build_prompt()),
                    ("human", message),
                ],
                CapabilityChoice,
            )
            if choice.key in {c.key for c in self._capabilities}:
                return choice.key
            return None
        except Exception:
            # LLM недоступен — детерминированный fallback по ключевым словам
            return self._keyword_classify(message)

    def _build_prompt(self) -> str:
        lines = "\n".join(
            f"- {c.key} — {c.description}" for c in self._capabilities
        )
        return (
            "Ты — маршрутизатор личного агента поиска работы. "
            "Доступные возможности:\n"
            f"{lines}\n"
            "Верни ключ одной из возможностей. Если это обычный разговор или "
            "вопрос — верни chat. Если пользователь просит выполнить действие, "
            'которого нет в списке, — верни "none".'
        )

    def _keyword_classify(self, message: str) -> str | None:
        lower = message.lower()
        for cap in self._capabilities:
            if cap.key == "chat":
                continue
            if _has_keyword(lower, _KEYWORDS.get(cap.key, ())):
                return cap.key
        return "chat"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_capabilities.py -q`
Expected: `4 passed`

- [ ] **Step 5: Delete `tests/test_router.py` (now obsolete)**

Run: `rm tests/test_router.py`
Note: `tests/test_router.py` tested `keyword_classify` from `src.agent.router`, which is removed in Task 6.

- [ ] **Step 6: Commit**

```bash
git add src/agent/capabilities.py tests/test_capabilities.py tests/test_router.py
git commit -m "feat: add capability registry router"
```

---

## Task 6: Refactor `build_graph` to search-only pipeline + RAG

**Files:**
- Modify: `src/agent/graph.py`
- Delete: `src/agent/router.py` (no longer used)
- Test: `tests/test_agent_graph.py` (updated in Task 7)

- [ ] **Step 1: Rewrite `src/agent/graph.py`**

Replace the entire file with:

```python
from typing import Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from src.agent.policy import requires_human_approval
from src.agent.state import AgentState
from src.core.audit import log_action
from src.llm.gateway import LLMGateway


class CandidateScore(BaseModel):
    """Оценка соответствия вакансии профилю пользователя."""

    score: float  # 0..1
    reason: str


# Порог: вакансии с оценкой выше идут в отклик
APPLY_THRESHOLD = 0.6


async def _score_candidate(
    gateway: LLMGateway, candidate: dict, profile: dict, chunks: list[str] | None = None
) -> CandidateScore:
    """Оценить соответствие вакансии профилю через LLM (структурированный вывод).

    :param gateway: единая точка доступа к LLM
    :param candidate: словарь с данными вакансии
    :param profile: профиль соискателя (навыки/опыт) для сопоставления
    :param chunks: релевантные фрагменты резюме (RAG), подмешиваемые в промпт
    :return: оценка соответствия (score + reason)
    """
    chunks = chunks or []
    context = f"Профиль: {profile}"
    if chunks:
        fragments = "\n".join(f"- {chunk}" for chunk in chunks)
        context += f"\nРелевантные фрагменты резюме:\n{fragments}"
    return await gateway.invoke_structured(
        gateway.text_model,
        [
            (
                "system",
                "Оцени релевантность вакансии профилю соискателя от 0 до 1. "
                "score — число, reason — краткое обоснование.",
            ),
            ("human", f"{context}\nВакансия: {candidate}"),
        ],
        CandidateScore,
    )


async def _retrieve_chunks(
    user_id: int,
    candidate: dict,
    retriever: Callable[[int, str], Awaitable[list[str]]] | None,
) -> list[str]:
    """Извлечь релевантные фрагменты резюме; при сбое RAG — пустой список."""
    if retriever is None:
        return []
    query = f"{candidate.get('title', '')} {candidate.get('description', '')}".strip()
    if not query:
        return []
    try:
        return await retriever(user_id, query)
    except Exception:
        # RAG недоступен (нет эмбеддингов) — скорим без фрагментов
        return []


def build_graph(
    gateway: LLMGateway,
    searcher: Callable[[int, str], Awaitable[list[dict]]],
    checkpointer: BaseCheckpointSaver | None = None,
    applier: Callable[[int, dict], Awaitable[None]] | None = None,
    profile_provider: Callable[[int], Awaitable[dict]] | None = None,
    retriever: Callable[[int, str], Awaitable[list[str]]] | None = None,
):
    """Собрать граф поиска: search → match → decision → apply → report + HITL.

    :param gateway: LLMGateway (скоринг, письма)
    :param searcher: async callable(user_id, query) -> list[dict]
    :param checkpointer: checkpointer LangGraph (для HITL)
    :param applier: async callable(user_id, decision) — реальный отклик
    :param profile_provider: async callable(user_id) -> dict — профиль соискателя
    :param retriever: async callable(user_id, query) -> list[str] — фрагменты резюме
    :return: скомпилированный граф
    """

    async def search_node(state: AgentState) -> dict:
        """Найти вакансии через внедрённый searcher по запросу пользователя."""
        query = state.get("user_message", "")
        candidates = await searcher(state["user_id"], query)
        return {"candidates": candidates}

    async def match_node(state: AgentState) -> dict:
        """Оценить каждую вакансию через LLM-скоринг с учётом профиля и RAG."""
        profile = await profile_provider(state["user_id"]) if profile_provider else {}
        decisions = []
        for c in state.get("candidates", []):
            chunks = await _retrieve_chunks(state["user_id"], c, retriever)
            scored = await _score_candidate(gateway, c, profile, chunks)
            decisions.append({"job": c, "score": scored.score, "reason": scored.reason})
        return {"decisions": decisions}

    async def decision_node(state: AgentState) -> dict:
        """Вынести решение apply/skip по порогу релевантности и набросать письмо."""
        final = []
        for d in state.get("decisions", []):
            decision = "apply" if d["score"] >= APPLY_THRESHOLD else "skip"
            d["decision"] = decision
            if decision == "apply":
                # Черновик сопроводительного письма под конкретную вакансию
                msg = await gateway.text_model.ainvoke(
                    [
                        (
                            "system",
                            "Ты пишешь краткое сопроводительное письмо на русском "
                            "(3-5 предложений) под конкретную вакансию.",
                        ),
                        ("human", f"Вакансия: {d['job']}"),
                    ]
                )
                d["cover_letter"] = msg.content
            else:
                d["cover_letter"] = ""
            final.append(d)
        return {"decisions": final}

    async def apply_node(state: AgentState) -> dict:
        """Откликнуться на подходящие вакансии с HITL-подтверждением."""
        applied = []
        for d in state.get("decisions", []):
            if d.get("decision") != "apply":
                continue
            # Высокорисковое действие: пауза до подтверждения человека
            if requires_human_approval("apply"):
                approval = interrupt(
                    {
                        "action": "apply",
                        "job": d.get("job"),
                        "cover_letter": d.get("cover_letter", ""),
                    }
                )
                # Аудит: что агент хотел сделать и какое решение принято
                log_action(
                    state["user_id"],
                    "apply",
                    "approved" if approval is True else "rejected",
                    {"job": d.get("job", {})},
                )
                if approval is not True:
                    continue
            # Реальное выполнение отклика через внедрённый side-effect
            if applier is not None:
                await applier(state["user_id"], d)
            applied.append(d)
        return {"decisions": applied, "needs_human": False, "pending_action": None}

    async def report_node(state: AgentState) -> dict:
        """Сформировать итоговую статистику и текст ответа пользователю."""
        applied_count = len(state.get("decisions", []))
        report = {"applied_count": applied_count, "replied_count": 0}
        reply = f"Откликнулся на {applied_count} вакансий"
        return {"report": report, "reply": reply}

    graph = StateGraph(AgentState)
    graph.add_node("search", search_node)
    graph.add_node("match", match_node)
    graph.add_node("decision", decision_node)
    graph.add_node("apply", apply_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "match")
    graph.add_edge("match", "decision")
    graph.add_edge("decision", "apply")
    graph.add_edge("apply", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 2: Delete `src/agent/router.py`**

Run: `rm src/agent/router.py`

- [ ] **Step 3: Verify nothing else imports the router**

Run: `uv run python -c "import src.agent.graph" && rg -n "from src.agent.router|agent.router import" src tests`
Expected: `import src.agent.graph` succeeds (after the two test files are updated in Tasks 7/11); remaining `router` imports surface here and are fixed in the next tasks.

- [ ] **Step 4: Commit (tests still broken until Task 7 — commit graph refactor alone is acceptable, but prefer squashing with Task 7)**

Skip standalone commit; commit together with Task 7.

---

## Task 7: Update `tests/test_agent_graph.py`

**Files:**
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Rewrite `tests/test_agent_graph.py`**

Replace the entire file with:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agent.graph import CandidateScore, build_graph


class FakeGateway:
    """Поддельный gateway: возвращает предзаданный скоринг, фиксирует вызовы."""

    def __init__(self, score: float = 0.1) -> None:
        self.score = score
        self.text_model = _FakeModel()
        self.calls: list[tuple[str, list]] = []

    async def invoke_structured(self, model, messages, schema):
        name = schema.__name__
        self.calls.append((name, messages))
        if name == "CandidateScore":
            return CandidateScore(score=self.score, reason="ok")
        raise ValueError(f"unexpected schema {name}")


class _FakeModel:
    async def ainvoke(self, messages):
        return type("Msg", (), {"content": "привет, чем помочь?"})()


class FakeSearcher:
    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    async def __call__(self, user_id: int, query: str = "") -> list[dict]:
        return self._candidates


def _build(gateway: FakeGateway, searcher: FakeSearcher | None = None, **kwargs):
    """Собрать граф с фейками и чекпоинтером для HITL-тестов."""
    searcher = searcher or FakeSearcher([])
    return build_graph(gateway, searcher, checkpointer=MemorySaver(), **kwargs)


async def test_search_flow_no_applies():
    """Низкий скоринг: все вакансии skip, откликов нет и интеррапта нет."""
    graph = _build(FakeGateway(score=0.1), FakeSearcher([{"title": "Dev"}]))
    result = await graph.ainvoke(
        {"user_id": 1, "user_message": "найди работу"},
        {"configurable": {"thread_id": "t2"}},
    )
    assert result["report"]["applied_count"] == 0


async def test_hitl_apply_requires_approval():
    """Высокий скоринг: граф останавливается на интеррапте и ждёт подтверждения."""
    graph = _build(FakeGateway(score=0.9), FakeSearcher([{"title": "Dev"}]))
    config = {"configurable": {"thread_id": "t3"}}

    result = await graph.ainvoke({"user_id": 1, "user_message": "найди работу"}, config)
    assert result.get("report") is None
    assert "__interrupt__" in result

    resumed = await graph.ainvoke(Command(resume=True), config)
    assert resumed["report"]["applied_count"] == 1


async def test_apply_calls_applier_after_approval():
    """После подтверждения HITL applier вызывается с откликом и письмом."""
    calls: list[tuple[int, dict]] = []

    async def applier(user_id: int, decision: dict) -> None:
        calls.append((user_id, decision))

    graph = _build(
        FakeGateway(score=0.9),
        FakeSearcher([{"title": "Dev"}]),
        applier=applier,
    )
    config = {"configurable": {"thread_id": "t4"}}

    await graph.ainvoke({"user_id": 1, "user_message": "найди работу"}, config)
    resumed = await graph.ainvoke(Command(resume=True), config)
    assert resumed["report"]["applied_count"] == 1

    assert len(calls) == 1
    user_id, decision = calls[0]
    assert user_id == 1
    assert decision["cover_letter"]


async def test_match_uses_profile():
    """Профиль соискателя попадает в промпт скоринга вакансии."""

    async def profile_provider(user_id: int) -> dict:
        return {"skills": ["python"]}

    gateway = FakeGateway(score=0.1)
    graph = _build(
        gateway,
        FakeSearcher([{"title": "Dev"}]),
        profile_provider=profile_provider,
    )
    await graph.ainvoke(
        {"user_id": 1, "user_message": "найди работу"},
        {"configurable": {"thread_id": "t5"}},
    )

    for name, messages in gateway.calls:
        if name != "CandidateScore":
            continue
        human = [m[1] for m in messages if m[0] == "human"][0]
        assert "python" in human
        return
    raise AssertionError("CandidateScore не вызывался")


async def test_match_uses_rag_chunks():
    """Релевантные фрагменты резюме попадают в промпт скоринга."""

    async def retriever(user_id: int, query: str) -> list[str]:
        return ["Опыт: 5 лет Python", "Навыки: FastAPI"]

    gateway = FakeGateway(score=0.1)
    graph = _build(
        gateway,
        FakeSearcher([{"title": "Python Dev"}]),
        retriever=retriever,
    )
    await graph.ainvoke(
        {"user_id": 1, "user_message": "найди работу"},
        {"configurable": {"thread_id": "t6"}},
    )

    for name, messages in gateway.calls:
        if name != "CandidateScore":
            continue
        human = [m[1] for m in messages if m[0] == "human"][0]
        assert "Опыт: 5 лет Python" in human
        assert "Навыки: FastAPI" in human
        return
    raise AssertionError("CandidateScore не вызывался")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_graph.py -q`
Expected: `5 passed`

- [ ] **Step 3: Commit (together with Task 6)**

```bash
git add src/agent/graph.py src/agent/router.py tests/test_agent_graph.py
git commit -m "refactor: shrink graph to search pipeline, add RAG retriever"
```

---

## Task 8: Add migration to align vector dimension to 768

**Files:**
- Create: `migrations/versions/<hash>_align_embedding_dim.py` (generated by alembic, then edited)

- [ ] **Step 1: Generate the migration**

Run: `uv run alembic revision -m "align embedding dim to 768"`
Expected: a new file appears under `migrations/versions/`.

- [ ] **Step 2: Edit the generated file**

Open the new file and replace the `upgrade`/`downgrade` bodies with raw SQL (keep the
generated `revision`/`down_revision`/imports):

```python
def upgrade() -> None:
    op.execute(
        "ALTER TABLE documents ALTER COLUMN embedding "
        "TYPE vector(768) USING embedding::vector(768)"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE documents ALTER COLUMN embedding "
        "TYPE vector(1536) USING embedding::vector(1536)"
    )
```

- [ ] **Step 3: Apply the migration to the local DB**

Run: `uv run alembic upgrade head`
Expected: `INFO [alembic.runtime.migration] Running upgrade ... -> <hash>, align embedding dim to 768`

- [ ] **Step 4: Commit**

```bash
git add migrations/versions/<hash>_align_embedding_dim.py
git commit -m "db: align documents.embedding dimension to 768"
```

---

## Task 9: Add dispatcher + `handle_document` to `TgService`

**Files:**
- Modify: `src/tg/service.py`
- Test: `tests/test_tg.py` (updated in Task 11)

- [ ] **Step 1: Rewrite `src/tg/service.py`**

Replace the entire file with:

```python
import logging
import re
import tempfile
from pathlib import Path

from langgraph.types import Command

from src.agent.capabilities import Capability, CapabilityRouter
from src.agent.sites import SitesRepository, parse_sites_message
from src.rag.ingest import extract_cv_text
from src.rag.retrieve import extract_profile, ingest_resume_chunks

logger = logging.getLogger(__name__)

# Точные слова-подтверждения HITL-действия
_APPROVE_WORDS = (
    "да",
    "yes",
    "ок",
    "окей",
    "подтверждаю",
    "согласен",
    "отправляй",
    "давай",
)

# Слова-отказы: обрабатываются в первую очередь, чтобы «не отправляй» не считалось одобрением
_REJECT_WORDS = ("нет", "no", "не ", "стоп", "отмена", "откаж", "не надо")

# Поддерживаемые форматы резюме
_RESUME_EXTENSIONS = {".pdf", ".docx"}


def _has_keyword(lower: str, words: tuple[str, ...]) -> bool:
    """Проверить вхождение ключевого слова по границе слова."""
    return any(re.search(rf"\b{re.escape(word)}", lower) for word in words)


class TgService:
    """Оркестрация диалога: сообщение → способность → текст ответа."""

    def __init__(
        self,
        graph,
        session_factory,
        gateway=None,
        sites_repo=None,
        profile_repo=None,
        stats_provider=None,
    ) -> None:
        self._graph = graph
        self._session_factory = session_factory
        self._gateway = gateway
        self._sites = sites_repo or SitesRepository(session_factory)
        self._profile_repo = profile_repo
        self._stats_provider = stats_provider

        # === Реестр способностей: метаданные для LLM + сопоставление с обработчиками ===
        self._capabilities = [
            Capability(key="upload_resume", description="принять и разобрать резюме (PDF/DOCX)"),
            Capability(key="search_job", description="искать и откликаться на вакансии"),
            Capability(key="stats", description="показать статистику откликов"),
            Capability(key="chat", description="свободная переписка, ответы на вопросы"),
        ]
        self._router = CapabilityRouter(gateway, self._capabilities)
        self._handlers = {
            "search_job": self._handle_search,
            "stats": self._handle_stats,
            "upload_resume": self._handle_upload_resume,
            "chat": self._handle_chat,
        }

    async def handle_message(self, telegram_id: str, text: str) -> str:
        """Обработать входящее текстовое сообщение и вернуть ответ.

        :param telegram_id: идентификатор пользователя в Telegram
        :param text: текст сообщения
        :return: текст ответа для отправки в Telegram
        """
        user_id = await self._get_or_create_user(telegram_id)

        # Onboarding: без сайтов просим назвать площадки и не запускаем диспетчер
        domains = await self._sites.get_domains(user_id)
        if not domains:
            parsed = await parse_sites_message(self._gateway, text)
            if parsed:
                await self._sites.add_domains(user_id, parsed)
                return f"Запомнил. Буду искать на: {', '.join(parsed)}."
            return "На каких сайтах искать работу? Назови сайты через запятую."

        config = {"configurable": {"thread_id": f"tg-{telegram_id}"}}

        # Если граф приостановлен (ожидает HITL) — возобновляем с решением пользователя
        snapshot = await self._graph.aget_state(config)
        if snapshot is not None and snapshot.next:
            result = await self._graph.ainvoke(
                Command(resume=self._is_approval(text)), config
            )
            return self._build_reply(result)

        key = await self._router.classify(text)
        handler = self._handlers.get(key)
        if handler is None:
            return self._not_available_reply()
        return await handler(user_id, text, config)

    async def handle_document(
        self, telegram_id: str, filename: str, content: bytes
    ) -> str:
        """Обработать присланный документ как резюме и сохранить профиль + чанки.

        :param telegram_id: идентификатор пользователя в Telegram
        :param filename: исходное имя файла
        :param content: байты файла
        :return: текст ответа для отправки в Telegram
        """
        user_id = await self._get_or_create_user(telegram_id)

        suffix = Path(filename).suffix.lower()
        if suffix not in _RESUME_EXTENSIONS:
            return "Пришли резюме в формате PDF или DOCX."

        text = await self._read_cv_text(suffix, content)
        if not text.strip():
            return "Не удалось прочитать документ."

        # Структурированный профиль из текста CV
        profile = await extract_profile(self._gateway, text)
        if self._profile_repo is not None:
            await self._profile_repo.save(user_id, profile)

        # Чанки + эмбеддинги для RAG; при сбое — профиль уже сохранён
        try:
            await ingest_resume_chunks(
                self._session_factory, self._gateway.embedder, user_id, text
            )
        except Exception as exc:  # noqa: BLE001 - RAG не должен ронять загрузку CV
            logger.warning("RAG ingest failed for user %s: %s", user_id, exc)

        return (
            f"Записал профиль: роль — {profile.desired_role or '—'}, "
            f"навыков — {len(profile.skills)}."
        )

    async def _read_cv_text(self, suffix: str, content: bytes) -> str:
        """Извлечь текст из байтов документа через временный файл."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            return await extract_cv_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _handle_search(self, user_id: int, text: str, config: dict) -> str:
        """Способность search_job: запустить граф поиска и собрать ответ."""
        result = await self._graph.ainvoke(
            {"user_id": user_id, "user_message": text}, config
        )
        return self._build_reply(result)

    async def _handle_stats(self, user_id: int, text: str, config: dict) -> str:
        """Способность stats: вернуть накопленную статистику из БД."""
        if self._stats_provider is None:
            return "Статистика недоступна"
        stats = await self._stats_provider(user_id)
        return (
            f"Всего откликов: {stats.get('applied_count', 0)}, "
            f"общений с работодателями: {stats.get('replied_count', 0)}"
        )

    async def _handle_upload_resume(self, user_id: int, text: str, config: dict) -> str:
        """Способность upload_resume: попросить прислать файл."""
        return "Пришли резюме в формате PDF или DOCX сюда."

    async def _handle_chat(self, user_id: int, text: str, config: dict) -> str:
        """Способность chat: свободный ответ через текстовую модель."""
        reply_msg = await self._gateway.text_model.ainvoke(
            [
                (
                    "system",
                    "Ты — дружелюбный ассистент по поиску работы. "
                    "Отвечай кратко на русском.",
                ),
                ("human", text),
            ]
        )
        return reply_msg.content

    def _not_available_reply(self) -> str:
        """Ответ, когда ни одна способность не подошла."""
        abilities = ", ".join(c.description for c in self._capabilities)
        return f"Такого функционала пока нет. Я умею: {abilities}."

    async def _get_or_create_user(self, telegram_id: str) -> int:
        """Найти или создать пользователя по telegram_id."""
        from sqlalchemy import select

        from src.database.models import User

        async with self._session_factory() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            if user is None:
                user = User(telegram_id=telegram_id)
                session.add(user)
                await session.commit()
                await session.refresh(user)
            return user.id

    def _is_approval(self, text: str) -> bool:
        """Определить, является ли сообщение подтверждением действия."""
        lower = text.lower().strip()
        if _has_keyword(lower, _REJECT_WORDS):
            return False
        return _has_keyword(lower, _APPROVE_WORDS)

    def _build_reply(self, result: dict) -> str:
        """Сформировать текст ответа из результата работы графа."""
        interrupts = result.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value
            job = pending.get("job", {})
            title = job.get("title", "вакансию")
            return f"Подтвердить отклик на «{title}»? Ответь «да» или «нет»."
        return result.get("reply", "Готово")
```

- [ ] **Step 2: Commit (tests broken until Task 11; defer commit)**

Skip standalone commit; commit together with Task 11.

---

## Task 10: Update `src/tg/bot.py` composition root + document handler

**Files:**
- Modify: `src/tg/bot.py`

- [ ] **Step 1: Rewrite `build_tg_service` and add the document handler**

In `src/tg/bot.py`:

1. Add `from aiogram import F` to the aiogram import line (line 4).
2. Add imports for `ProfileRepository` and `ResumeRetriever`.
3. Replace `build_tg_service` (lines 53-83) with:

```python
def build_tg_service() -> TgService:
    """Собрать TgService со всеми зависимостями (композиционный корень)."""
    from langgraph.checkpoint.memory import MemorySaver

    gateway = LLMGateway()
    browser = BrowserExecutor(settings.BROWSER_EXECUTOR_URL)
    sites_repo = SitesRepository(SessionLocal)
    profile_repo = ProfileRepository(SessionLocal)
    searcher = UniversalSearcher(
        browser, gateway, sites_repo, profile_provider=_get_profile
    )
    applier = UniversalApplier(browser, gateway, sites_repo)
    retriever = ResumeRetriever(SessionLocal, gateway.embedder)

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
        checkpointer=MemorySaver(),
        applier=_apply,
        profile_provider=_get_profile,
        retriever=retriever,
    )
    return TgService(
        graph,
        SessionLocal,
        gateway=gateway,
        sites_repo=sites_repo,
        profile_repo=profile_repo,
        stats_provider=_get_stats,
    )
```

4. Add a document handler in `main()` after the `start` handler (before the generic
   `@dp.message()`):

```python
    @dp.message(F.content_type == "document")
    async def on_document(message: Message) -> None:
        document = message.document
        file = await bot.get_file(document.file_id)
        downloaded = await bot.download_file(file.file_path)
        reply = await service.handle_document(
            str(message.from_user.id), document.file_name, downloaded.read()
        )
        await message.answer(reply)
```

5. Add the new imports near the top (after existing `from src.stats.aggregate import ...`):

```python
from src.rag.profile_repo import ProfileRepository
from src.rag.retrieve import ResumeRetriever
```

- [ ] **Step 2: Verify the module imports**

Run: `uv run python -c "import src.tg.bot"`
Expected: no error.

- [ ] **Step 3: Commit (together with Task 11)**

---

## Task 11: Update `tests/test_tg.py` and add dispatcher + document tests

**Files:**
- Modify: `tests/test_tg.py`

- [ ] **Step 1: Rewrite `tests/test_tg.py`**

Replace the entire file with:

```python
from langgraph.checkpoint.memory import MemorySaver

from src.agent.capabilities import CapabilityChoice
from src.agent.graph import CandidateScore, build_graph
from src.rag.schemas import ProfileData
from src.tg.service import TgService


class FakeModel:
    """Поддельная текстовая модель: фиксированный ответ на любое сообщение."""

    async def ainvoke(self, messages):
        return type("Msg", (), {"content": "привет, чем помочь?"})()


class FakeGateway:
    """Поддельный gateway: возвращает предзаданные структурированные ответы."""

    def __init__(self, capability_key: str | None = "chat", score: float = 0.1) -> None:
        self.capability_key = capability_key
        self.score = score
        self.text_model = FakeModel()

    async def invoke_structured(self, model, messages, schema):
        name = schema.__name__
        if name == "CapabilityChoice":
            return CapabilityChoice(key=self.capability_key)
        if name == "CandidateScore":
            return CandidateScore(score=self.score, reason="ok")
        if name == "ProfileData":
            return ProfileData(skills=["Python"], desired_role="разработчик")
        raise ValueError(f"unexpected schema {name}")


class FakeSearcher:
    """Поддельный поиск: возвращает заранее заданный список вакансий."""

    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    async def __call__(self, user_id: int, query: str = "") -> list[dict]:
        return self._candidates


class _FakeResult:
    """Поддельный результат запроса: пользователь не найден."""

    def scalar_one_or_none(self):
        return None


class _FakeSession:
    """Поддельная асинхронная сессия: все операции — no-op, фиксирует add."""

    def __init__(self) -> None:
        self.added = []

    async def execute(self, *args, **kwargs):
        return _FakeResult()

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        pass

    async def refresh(self, obj) -> None:
        pass

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def _fake_session_factory():
    """Фабрика поддельных сессий: всегда создаёт нового пользователя."""

    def factory():
        return _FakeSession()

    return factory


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


class FakeProfileRepo:
    """Поддельный репозиторий профиля: фиксирует вызовы save."""

    def __init__(self) -> None:
        self.saved: list[tuple[int, ProfileData]] = []

    async def save(self, user_id: int, data: ProfileData) -> None:
        self.saved.append((user_id, data))


def _build_service(
    gateway: FakeGateway,
    searcher: FakeSearcher | None = None,
    sites_repo: FakeSitesRepo | None = None,
    profile_repo: FakeProfileRepo | None = None,
) -> TgService:
    """Собрать TgService с фейками и in-memory чекпоинтером."""
    searcher = searcher or FakeSearcher([])
    sites_repo = sites_repo if sites_repo is not None else FakeSitesRepo(["hh.ru"])
    graph = build_graph(gateway, searcher, checkpointer=MemorySaver())
    return TgService(
        graph,
        _fake_session_factory(),
        gateway=gateway,
        sites_repo=sites_repo,
        profile_repo=profile_repo,
    )


async def test_chat_message_returns_reply():
    """Свободное сообщение: способность chat, возвращает текст ответа."""
    service = _build_service(FakeGateway(capability_key="chat"))
    reply = await service.handle_message("123", "привет")
    assert reply == "привет, чем помочь?"


async def test_search_message_triggers_hitl():
    """Высокий скоринг: первый вызов возвращает запрос на подтверждение отклика."""
    service = _build_service(
        FakeGateway(capability_key="search_job", score=0.9),
        FakeSearcher([{"title": "Python Dev"}]),
    )
    reply = await service.handle_message("123", "найди работу")
    assert "Подтвердить отклик" in reply
    assert "Python Dev" in reply


def test_is_approval_rejects_negation():
    """Сообщения с отрицанием («не отправляй») не считаются подтверждением."""
    service = _build_service(FakeGateway(capability_key="chat"))
    assert service._is_approval("да") is True
    assert service._is_approval("не отправляй") is False
    assert service._is_approval("нет") is False


def test_is_approval_does_not_match_inside_words():
    """Короткие слова («да», «ок») не должны совпадать внутри других слов."""
    service = _build_service(FakeGateway(capability_key="chat"))
    assert service._is_approval("подарок") is False
    assert service._is_approval("свидание") is False
    assert service._is_approval("блок") is False
    assert service._is_approval("интернет") is False


async def test_confirm_resumes_and_reports():
    """Подтверждение «да» возобновляет приостановленный граф и даёт итоговый отчёт."""
    service = _build_service(
        FakeGateway(capability_key="search_job", score=0.9),
        FakeSearcher([{"title": "Python Dev"}]),
    )
    await service.handle_message("123", "найди работу")
    reply = await service.handle_message("123", "да")
    assert reply == "Откликнулся на 1 вакансий"


async def test_onboarding_asks_for_sites_when_none():
    """Нет сайтов: бот спрашивает, на каких сайтах искать, и не запускает диспетчер."""
    service = _build_service(FakeGateway(capability_key="chat"), sites_repo=FakeSitesRepo([]))
    reply = await service.handle_message("123", "привет")
    assert "На каких сайтах" in reply


async def test_onboarding_saves_sites_from_message():
    """Сообщение с доменами при отсутствии сайтов сохраняет их и подтверждает."""
    sites_repo = FakeSitesRepo([])
    service = _build_service(FakeGateway(capability_key="chat"), sites_repo=sites_repo)
    reply = await service.handle_message("123", "ищи на hh.ru и habr.com")
    assert "hh.ru" in reply
    assert "habr.com" in reply
    assert "hh.ru" in sites_repo.added
    assert "habr.com" in sites_repo.added


async def test_unknown_request_replies_not_available():
    """Запрос без подходящей способности → честный ответ «не умею»."""
    service = _build_service(FakeGateway(capability_key=None))
    reply = await service.handle_message("123", "отправь мне письмо на почту")
    assert "Такого функционала пока нет" in reply


async def test_upload_resume_prompts_file():
    """Способность upload_resume просит прислать файл."""
    service = _build_service(FakeGateway(capability_key="upload_resume"))
    reply = await service.handle_message("123", "куда прислать резюме")
    assert "PDF или DOCX" in reply


async def test_handle_document_saves_profile(tmp_path):
    """Присланный DOCX парсится и сохраняется в репозиторий профиля."""
    import docx

    doc = docx.Document()
    doc.add_paragraph("Иван Иванов")
    doc.add_paragraph("Python-разработчик")
    path = tmp_path / "cv.docx"
    doc.save(str(path))

    profile_repo = FakeProfileRepo()
    service = _build_service(
        FakeGateway(capability_key="chat"), profile_repo=profile_repo
    )
    reply = await service.handle_document(
        "123", "cv.docx", path.read_bytes()
    )

    assert "Записал профиль" in reply
    assert len(profile_repo.saved) == 1
    user_id, data = profile_repo.saved[0]
    assert user_id == 1
    assert data.skills == ["Python"]


async def test_handle_document_rejects_bad_format(tmp_path):
    """Неподдерживаемый формат отклоняется без парсинга."""
    service = _build_service(FakeGateway(capability_key="chat"))
    reply = await service.handle_document("123", "cv.txt", b"hello")
    assert "PDF или DOCX" in reply
```

Note: `test_handle_document_saves_profile` calls `handle_document`, which invokes
`ingest_resume_chunks(..., self._gateway.embedder, ...)`. `FakeGateway` has no
`embedder` attribute → `AttributeError` is raised and caught by the `try/except`,
so the test passes without touching a real embedder.

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_tg.py -q`
Expected: `12 passed`

- [ ] **Step 3: Commit (together with Tasks 9-10)**

```bash
git add src/tg/service.py src/tg/bot.py tests/test_tg.py
git commit -m "feat: capability dispatcher + resume document upload"
```

---

## Task 12: Full verification (tests + lint + format)

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass (no failures). The previously-failing
`test_store_and_retrieve_relevant` is fixed by Task 1.

- [ ] **Step 2: Run black lint check**

Run: `uv run black --check src tests config.py main.py`
Expected: `All done! ✨` (or fix formatting with `uv run black src tests config.py main.py` and re-run).

- [ ] **Step 3: Verify migrations apply cleanly on a fresh DB (optional, local)**

Run: `uv run alembic upgrade head`
Expected: no errors; final revision is `align embedding dim to 768`.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "style: black formatting"
```

---

## Self-review notes

- Spec coverage: capability registry (Task 5/9), resume upload (Task 3/4/9/10/11),
  RAG scoring (Task 2/4/6), vector dimension migration (Task 8), "не умею" (Task 9/11),
  keyword fallback (Task 5). All requirements covered.
- Type consistency: handler signature everywhere is
  `async (user_id: int, text: str, config: dict) -> str`; `Capability` has
  `key` + `description`; `CapabilityRouter.classify` returns `str | None`;
  `retriever` is `Callable[[int, str], Awaitable[list[str]]]`.
- Placeholder scan: the migration filename uses `<hash>` because alembic generates it;
  this is intentional and described in Task 8.
