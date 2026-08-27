from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agent.graph import CandidateScore, ResumeChoice, build_graph


class FakeGateway:
    """Поддельный gateway: возвращает предзаданный скоринг, фиксирует вызовы."""

    def __init__(self, score: float = 0.1, resume: str = "Python разработчик") -> None:
        self.score = score
        self.resume = resume
        self.text_model = _FakeModel()
        self.calls: list[tuple[str, list]] = []

    async def invoke_structured(self, model, messages, schema):
        name = schema.__name__
        self.calls.append((name, messages))
        if name == "CandidateScore":
            return CandidateScore(score=self.score, reason="ok")
        if name == "ResumeChoice":
            return ResumeChoice(resume=self.resume)
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


async def test_decision_uses_full_vacancy_text():
    """Полный текст вакансии попадает в промпт сопроводительного письма."""
    reads: list[str] = []

    async def vacancy_reader(user_id: int, url: str) -> str:
        reads.append(url)
        return "Требование: указать ожидаемую зарплату"

    captured: dict[str, list] = {}

    class _RecordingModel:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return type("Msg", (), {"content": "письмо"})()

    gateway = FakeGateway(score=0.9)
    gateway.text_model = _RecordingModel()

    graph = _build(
        gateway,
        FakeSearcher([{"title": "Dev", "url": "https://a.com/1"}]),
        vacancy_reader=vacancy_reader,
    )
    await graph.ainvoke(
        {"user_id": 1, "user_message": "найди работу"},
        {"configurable": {"thread_id": "t7"}},
    )

    assert reads == ["https://a.com/1"]
    human = [m[1] for m in captured["messages"] if m[0] == "human"][0]
    assert "Требование: указать ожидаемую зарплату" in human


async def test_decision_picks_resume():
    """Выбранное резюме попадает в решение и в HITL-подтверждение."""
    gateway = FakeGateway(score=0.9, resume="ML инженер")
    graph = _build(
        gateway,
        FakeSearcher([{"title": "ML Engineer"}]),
        resumes={"ML инженер": "ML/LLM", "Python разработчик": "backend"},
    )
    result = await graph.ainvoke(
        {"user_id": 1, "user_message": "найди работу"},
        {"configurable": {"thread_id": "t8"}},
    )

    pending = result["__interrupt__"][0].value
    assert pending["resume"] == "ML инженер"
