from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.router import Intent


class FakeGateway:
    """Поддельный gateway: возвращает предзаданные структурированные ответы."""

    def __init__(self, intent: str = "chat", score: float = 0.1) -> None:
        self.intent = intent
        self.score = score
        self.text_model = _FakeModel()
        self.calls: list[tuple[str, list]] = []

    async def invoke_structured(self, model, messages, schema):
        # Различаем схемы по имени класса
        name = schema.__name__
        self.calls.append((name, messages))
        if name == "Intent":
            return Intent(intent=self.intent)
        if name == "CandidateScore":
            from src.agent.graph import CandidateScore

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


def _build(gateway: FakeGateway, searcher: FakeSearcher | None = None):
    """Собрать граф с фейками и чекпоинтером для HITL-тестов."""
    searcher = searcher or FakeSearcher([])
    return build_graph(
        gateway,
        searcher,
        checkpointer=MemorySaver(),
    )


async def test_chat_flow():
    """Свободный чат: интент chat, ответ пользователю без поиска."""
    graph = _build(FakeGateway(intent="chat"))
    result = await graph.ainvoke(
        {"user_id": 1, "user_message": "привет"},
        {"configurable": {"thread_id": "t1"}},
    )
    assert result["intent"] == "chat"
    assert result["reply"]


async def test_search_flow_no_applies():
    """Низкий скоринг: все вакансии skip, откликов нет и интеррапта нет."""
    graph = _build(
        FakeGateway(intent="search_job", score=0.1),
        FakeSearcher([{"title": "Dev"}]),
    )
    result = await graph.ainvoke(
        {"user_id": 1, "user_message": "найди работу"},
        {"configurable": {"thread_id": "t2"}},
    )
    assert result["report"]["applied_count"] == 0


async def test_hitl_apply_requires_approval():
    """Высокий скоринг: граф останавливается на интеррапте и ждёт подтверждения."""
    graph = _build(
        FakeGateway(intent="search_job", score=0.9),
        FakeSearcher([{"title": "Dev"}]),
    )
    config = {"configurable": {"thread_id": "t3"}}

    result = await graph.ainvoke({"user_id": 1, "user_message": "найди работу"}, config)
    assert result.get("report") is None  # остановились до отчёта
    assert "__interrupt__" in result  # есть ожидающее HITL-подтверждение

    resumed = await graph.ainvoke(Command(resume=True), config)
    assert resumed["report"]["applied_count"] == 1


async def test_apply_calls_applier_after_approval():
    """После подтверждения HITL applier вызывается с откликом и письмом."""
    calls: list[tuple[int, dict]] = []

    async def applier(user_id: int, decision: dict) -> None:
        calls.append((user_id, decision))

    graph = build_graph(
        FakeGateway(intent="search_job", score=0.9),
        FakeSearcher([{"title": "Dev"}]),
        checkpointer=MemorySaver(),
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

    gateway = FakeGateway(intent="search_job", score=0.1)
    graph = build_graph(
        gateway,
        FakeSearcher([{"title": "Dev"}]),
        checkpointer=MemorySaver(),
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
