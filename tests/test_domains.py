import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.agent.sites import (
    Sites,
    SitesRepository,
    extract_domains,
    normalize_domain,
    parse_sites_message,
)
from src.database.db_settings import Base
from src.database.models import User


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


@pytest.fixture(scope="module")
def pg_url():
    try:
        from testcontainers.postgres import PostgresContainer

        container = PostgresContainer("pgvector/pgvector:pg16")
        container.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker недоступен: {exc}")
    url = container.get_connection_url(driver="asyncpg")
    yield url
    container.stop()


async def test_sites_repository_add_and_get(pg_url):
    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        session.add(User(id=1, telegram_id="t1"))
        await session.commit()

    repo = SitesRepository(session_factory)
    await repo.add_domains(1, ["hh.ru", "habr.com"])
    assert await repo.get_domains(1) == ["hh.ru", "habr.com"]

    await repo.add_domains(1, ["hh.ru"])
    assert await repo.get_domains(1) == ["hh.ru", "habr.com"]

    await engine.dispose()
