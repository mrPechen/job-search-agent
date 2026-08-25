import re

from langgraph.types import Command

from src.agent.sites import SitesRepository, parse_sites_message

# Точные слова-подтверждения HITL-действия
_APPROVE_WORDS = (
    "да",
    "yes",
    "ок",
    "окей",
    "подтверждаю",
    "согласен",
    "отправляй",
    "давай",
)

# Слова-отказы: обрабатываются в первую очередь, чтобы «не отправляй» не считалось одобрением
_REJECT_WORDS = ("нет", "no", "не ", "стоп", "отмена", "откаж", "не надо")


def _has_keyword(lower: str, words: tuple[str, ...]) -> bool:
    """Проверить вхождение ключевого слова по границе слова.

    Граница слова нужна, чтобы короткие слова («да», «ок») не совпадали
    внутри обычных слов («подарок», «блок»).
    """
    return any(re.search(rf"\b{re.escape(word)}", lower) for word in words)


class TgService:
    """Оркестрация диалога: сообщение пользователя → граф → текст ответа."""

    def __init__(self, graph, session_factory, gateway=None, sites_repo=None) -> None:
        self._graph = graph
        self._session_factory = session_factory
        self._gateway = gateway
        self._sites = sites_repo or SitesRepository(session_factory)

    async def handle_message(self, telegram_id: str, text: str) -> str:
        """Обработать входящее сообщение и вернуть текст ответа.

        :param telegram_id: идентификатор пользователя в Telegram
        :param text: текст сообщения
        :return: текст ответа для отправки в Telegram
        """
        user_id = await self._get_or_create_user(telegram_id)

        # Onboarding: без сайтов просим назвать площадки и не запускаем граф
        domains = await self._sites.get_domains(user_id)
        if not domains:
            parsed = await parse_sites_message(self._gateway, text)
            if parsed:
                await self._sites.add_domains(user_id, parsed)
                return f"Запомнил. Буду искать на: {', '.join(parsed)}."
            return "На каких сайтах искать работу? Назови сайты через запятую."

        config = {"configurable": {"thread_id": f"tg-{telegram_id}"}}

        # Если граф приостановлен (ожидает HITL) — возобновляем с решением пользователя
        snapshot = await self._graph.aget_state(config)
        if snapshot is not None and snapshot.next:
            result = await self._graph.ainvoke(
                Command(resume=self._is_approval(text)), config
            )
        else:
            result = await self._graph.ainvoke(
                {"user_id": user_id, "user_message": text}, config
            )

        return self._build_reply(result)

    async def _get_or_create_user(self, telegram_id: str) -> int:
        """Найти или создать пользователя по telegram_id.

        :param telegram_id: идентификатор пользователя в Telegram
        :return: внутренний id пользователя
        """
        from sqlalchemy import select

        from src.database.models import User

        async with self._session_factory() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            if user is None:
                user = User(telegram_id=telegram_id)
                session.add(user)
                await session.commit()
                await session.refresh(user)
            return user.id

    def _is_approval(self, text: str) -> bool:
        """Определить, является ли сообщение подтверждением действия.

        :param text: текст сообщения пользователя
        :return: True, если пользователь подтверждает действие
        """
        lower = text.lower().strip()
        # Отказ приоритетнее: «не отправляй» не должно считаться подтверждением
        if _has_keyword(lower, _REJECT_WORDS):
            return False
        return _has_keyword(lower, _APPROVE_WORDS)

    def _build_reply(self, result: dict) -> str:
        """Сформировать текст ответа из результата работы графа.

        :param result: результат ainvoke (содержит __interrupt__ при HITL-паузе)
        :return: текст ответа пользователю
        """
        # После interrupt в результате остаётся __interrupt__ — формируем вопрос на подтверждение
        interrupts = result.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value
            job = pending.get("job", {})
            title = job.get("title", "вакансию")
            return f"Подтвердить отклик на «{title}»? Ответь «да» или «нет»."
        return result.get("reply", "Готово")
