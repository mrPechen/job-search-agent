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
        1,
        "goal",
        SearchOutcome,
        allowed_domains=["example.com"],
        start_url="https://example.com",
    )
    assert len(outcome.candidates) == 1
    assert executor.calls[0] == ("navigate", "https://example.com", ["example.com"])
    assert ("click", "#search") in executor.calls
    assert ("type", "#search", "python") in executor.calls


class _RaisingGateway:
    def __init__(self):
        self.calls = 0
        self.vision_model = _FakeModel()

    async def invoke_structured(self, model, messages, schema):
        self.calls += 1
        raise RuntimeError("vlm down")


async def test_loop_returns_empty_when_vlm_fails():
    gateway = _RaisingGateway()
    loop = BrowserLoop(FakeExecutor(), gateway, max_steps=5)
    outcome = await loop.run(1, "goal", SearchOutcome, allowed_domains=[])
    assert outcome.candidates == []
    assert outcome.error == "vlm decision failed"
    assert gateway.calls == 2


async def test_loop_returns_empty_on_step_limit():
    actions = [BrowserAction(tool="click", args={"selector": "#x"})] * 100
    gateway = FakeLoopGateway(actions)
    executor = FakeExecutor()
    loop = BrowserLoop(executor, gateway, max_steps=2)
    outcome = await loop.run(1, "goal", SearchOutcome, allowed_domains=[])
    assert outcome.candidates == []
