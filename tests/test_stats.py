import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.database.db_settings import Base
from src.database.models import Application, Conversation, Job, JobSite, User
from src.stats.aggregate import get_user_stats


@pytest.fixture(scope="module")
def pg_url():
    """Поднять контейнер pgvector или пропустить тест, если Docker недоступен."""
    try:
        from testcontainers.postgres import PostgresContainer

        container = PostgresContainer("pgvector/pgvector:pg16")
        container.start()
    except Exception as exc:  # noqa: BLE001 - любая ошибка запуска = нет Docker
        pytest.skip(f"Docker недоступен: {exc}")
    url = container.get_connection_url(driver="asyncpg")
    yield url
    container.stop()


async def test_get_user_stats(pg_url):
    engine = create_async_engine(pg_url)

    # Подготовка схемы: расширение vector + все таблицы Base
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Наполнение: пользователь, площадка и вакансия для FK-ссылок откликов
    async with session_factory() as session:
        session.add(User(id=1, telegram_id="t1"))
        session.add(JobSite(id=1, name="hh", adapter_key="hh"))
        await session.flush()

        session.add(
            Job(
                id=1,
                user_id=1,
                site_id=1,
                external_id="ext1",
                title="Python dev",
                url="https://example.com/1",
            )
        )
        await session.flush()

        # Два «сделанных» отклика и один черновик (pending) — не считается
        session.add(Application(id=1, job_id=1, user_id=1, status="submitted"))
        session.add(Application(id=2, job_id=1, user_id=1, status="offer"))
        session.add(Application(id=3, job_id=1, user_id=1, status="pending"))

        # Одно общение с работодателем и одно — чат с пользователем (не считается)
        session.add(
            Conversation(
                id=1,
                user_id=1,
                application_id=1,
                kind="employer",
            )
        )
        session.add(
            Conversation(
                id=2,
                user_id=1,
                application_id=2,
                kind="user_chat",
            )
        )
        await session.commit()

    async with session_factory() as session:
        stats = await get_user_stats(session, user_id=1)

    assert stats == {"applied_count": 2, "replied_count": 1}

    await engine.dispose()
