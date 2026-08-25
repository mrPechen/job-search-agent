import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

from config import settings
from src.agent.graph import build_graph
from src.agent.router import IntentRouter
from src.agent.sites import SitesRepository
from src.agent.searcher import UniversalApplier, UniversalSearcher
from src.browser.executor import BrowserExecutor
from src.core.audit import log_action
from src.database.db_settings import SessionLocal
from src.llm.gateway import LLMGateway
from src.stats.aggregate import get_user_stats
from src.tg.service import TgService

logger = logging.getLogger(__name__)


async def _get_stats(user_id: int) -> dict:
    """Получить статистику пользователя из БД (открывает сессию на запрос)."""
    async with SessionLocal() as session:
        return await get_user_stats(session, user_id)


async def _get_profile(user_id: int) -> dict:
    """Получить профиль соискателя из БД для скоринга вакансий.

    :param user_id: внутренний id пользователя
    :return: dict с навыками, опытом и желаемой ролью (или {} при отсутствии)
    """
    from sqlalchemy import select

    from src.database.models import Profile

    async with SessionLocal() as session:
        result = await session.execute(
            select(Profile).where(Profile.user_id == user_id)
        )
        profile = result.scalars().first()
        if profile is None:
            return {}
        return {
            "skills": profile.skills,
            "experience": profile.experience,
            "desired_role": profile.desired_role,
        }


def build_tg_service() -> TgService:
    """Собрать TgService со всеми зависимостями (композиционный корень)."""
    from langgraph.checkpoint.memory import MemorySaver

    gateway = LLMGateway()
    browser = BrowserExecutor(settings.BROWSER_EXECUTOR_URL)
    sites_repo = SitesRepository(SessionLocal)
    searcher = UniversalSearcher(
        browser, gateway, sites_repo, profile_provider=_get_profile
    )
    applier = UniversalApplier(browser, gateway, sites_repo)

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
        router=IntentRouter(gateway),
        checkpointer=MemorySaver(),
        get_stats=_get_stats,
        applier=_apply,
        profile_provider=_get_profile,
    )
    return TgService(graph, SessionLocal, gateway=gateway, sites_repo=sites_repo)


async def main() -> None:
    """Точка входа: поднять бота и запустить поллинг сообщений."""
    bot = Bot(token=settings.TG_BOT_TOKEN)
    dp = Dispatcher()
    service = build_tg_service()

    @dp.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(
            "Привет! Напиши, что ищем по работе — я найду вакансии и откликнусь."
        )

    @dp.message()
    async def on_message(message: Message) -> None:
        reply = await service.handle_message(
            str(message.from_user.id), message.text or ""
        )
        await message.answer(reply)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
