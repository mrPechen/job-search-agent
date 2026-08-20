import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

from config import settings
from src.agent.graph import build_graph
from src.agent.router import IntentRouter
from src.browser.executor import BrowserExecutor
from src.browser.searcher import HhSearcher
from src.database.db_settings import SessionLocal
from src.llm.gateway import LLMGateway
from src.tg.service import TgService

logger = logging.getLogger(__name__)


def build_tg_service() -> TgService:
    """Собрать TgService со всеми зависимостями (композиционный корень)."""
    from langgraph.checkpoint.memory import MemorySaver

    gateway = LLMGateway()
    browser = BrowserExecutor(settings.BROWSER_EXECUTOR_URL)
    searcher = HhSearcher(browser)
    graph = build_graph(
        gateway,
        searcher,
        router=IntentRouter(gateway),
        checkpointer=MemorySaver(),
    )
    return TgService(graph, SessionLocal)


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
