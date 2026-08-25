from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import build_graph
from src.agent.router import Intent
from src.tg.service import TgService


class FakeModel:
    """Поддельная текстовая модель: фиксированный ответ на любое сообщение."""

    async def ainvoke(self, messages):
        return type("Msg", (), {"content": "привет, чем помочь?"})()


class FakeGateway:
    """Поддельный gateway: возвращает предзаданные структурированные ответы."""

    def __init__(self, intent: str = "chat", score: float = 0.1) -> None:
        self.intent = intent
        self.score = score
        self.text_model = FakeModel()

    async def invoke_structured(self, model, messages, schema):
        # Различаем схемы по имени класса
        name = schema.__name__
        if name == "Intent":
            return Intent(intent=self.intent)
        if name == "CandidateScore":
            from src.agent.graph import CandidateScore

            return CandidateScore(score=self.score, reason="ok")
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


async def test_chat_message_returns_reply():
    """Свободное сообщение: граф в режиме чата и возвращает текст ответа."""
    service = _build_service(FakeGateway(intent="chat"))
    reply = await service.handle_message("123", "привет")
    assert reply == "привет, чем помочь?"


async def test_search_message_triggers_hitl():
    """Высокий скоринг: первый вызов возвращает запрос на подтверждение отклика."""
    service = _build_service(
        FakeGateway(intent="search_job", score=0.9),
        FakeSearcher([{"title": "Python Dev"}]),
    )
    reply = await service.handle_message("123", "найди работу")
    assert "Подтвердить отклик" in reply
    assert "Python Dev" in reply


def test_is_approval_rejects_negation():
    """Сообщения с отрицанием («не отправляй») не считаются подтверждением."""
    service = _build_service(FakeGateway(intent="chat"))
    assert service._is_approval("да") is True
    assert service._is_approval("не отправляй") is False
    assert service._is_approval("нет") is False


def test_is_approval_does_not_match_inside_words():
    """Короткие слова («да», «ок») не должны совпадать внутри других слов."""
    service = _build_service(FakeGateway(intent="chat"))
    assert service._is_approval("подарок") is False
    assert service._is_approval("свидание") is False
    assert service._is_approval("блок") is False
    assert service._is_approval("интернет") is False


async def test_confirm_resumes_and_reports():
    """Подтверждение «да» возобновляет приостановленный граф и даёт итоговый отчёт."""
    service = _build_service(
        FakeGateway(intent="search_job", score=0.9),
        FakeSearcher([{"title": "Python Dev"}]),
    )
    await service.handle_message("123", "найди работу")
    reply = await service.handle_message("123", "да")
    assert reply == "Откликнулся на 1 вакансий"


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
