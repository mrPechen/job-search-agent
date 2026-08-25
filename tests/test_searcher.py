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
