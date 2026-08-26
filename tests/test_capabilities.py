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
