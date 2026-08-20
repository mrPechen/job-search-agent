from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Application, Conversation

# Статусы отклика, которые означают, что отклик фактически сделан (не «черновик»)
APPLIED_STATUSES = ("submitted", "viewed", "replied", "interview", "offer")


async def get_user_stats(session: AsyncSession, user_id: int) -> dict:
    """Посчитать статистику пользователя.

    :param session: асинхронная сессия БД
    :param user_id: внутренний id пользователя
    :return: dict с полями applied_count (откликов) и replied_count (общений)
    """
    applied = await session.scalar(
        select(func.count())
        .select_from(Application)
        .where(
            Application.user_id == user_id,
            Application.status.in_(APPLIED_STATUSES),
        )
    )
    replied = await session.scalar(
        select(func.count())
        .select_from(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.kind == "employer",
        )
    )
    return {"applied_count": applied or 0, "replied_count": replied or 0}
