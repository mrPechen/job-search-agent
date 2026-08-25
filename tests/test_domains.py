from src.agent.sites import (
    Sites,
    extract_domains,
    normalize_domain,
    parse_sites_message,
)


def test_normalize_domain_strips_scheme_path_and_www():
    assert normalize_domain("https://hh.ru/search/vacancy") == "hh.ru"
    assert normalize_domain("www.habr.com") == "habr.com"
    assert normalize_domain("HH.RU") == "hh.ru"


def test_normalize_domain_rejects_junk():
    assert normalize_domain("") is None
    assert normalize_domain("не сайт") is None


def test_extract_domains_dedupes_in_order():
    assert extract_domains("ищи на hh.ru и hh.ru, и на habr.com") == [
        "hh.ru",
        "habr.com",
    ]


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
