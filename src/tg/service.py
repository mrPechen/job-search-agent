import logging
import re
import tempfile
from pathlib import Path

from langgraph.types import Command

from src.agent.capabilities import Capability, CapabilityRouter
from src.agent.sites import SitesRepository, parse_sites_message
from src.rag.ingest import extract_cv_text
from src.rag.retrieve import extract_profile, ingest_resume_chunks

logger = logging.getLogger(__name__)

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

# Поддерживаемые форматы резюме
_RESUME_EXTENSIONS = {".pdf", ".docx"}


def _has_keyword(lower: str, words: tuple[str, ...]) -> bool:
    """Проверить вхождение ключевого слова по границе слова."""
    return any(re.search(rf"\b{re.escape(word)}", lower) for word in words)


class TgService:
    """Оркестрация диалога: сообщение → способность → текст ответа."""

    def __init__(
        self,
        graph,
        session_factory,
        gateway=None,
        sites_repo=None,
        profile_repo=None,
        stats_provider=None,
    ) -> None:
        self._graph = graph
        self._session_factory = session_factory
        self._gateway = gateway
        self._sites = sites_repo or SitesRepository(session_factory)
        self._profile_repo = profile_repo
        self._stats_provider = stats_provider

        # === Реестр способностей: метаданные для LLM + сопоставление с обработчиками ===
        self._capabilities = [
            Capability(
                key="upload_resume", description="принять и разобрать резюме (PDF/DOCX)"
            ),
            Capability(
                key="search_job", description="искать и откликаться на вакансии"
            ),
            Capability(key="stats", description="показать статистику откликов"),
            Capability(
                key="chat", description="свободная переписка, ответы на вопросы"
            ),
        ]
        self._router = CapabilityRouter(gateway, self._capabilities)
        self._handlers = {
            "search_job": self._handle_search,
            "stats": self._handle_stats,
            "upload_resume": self._handle_upload_resume,
            "chat": self._handle_chat,
        }

    async def handle_message(self, telegram_id: str, text: str) -> str:
        """Обработать входящее текстовое сообщение и вернуть ответ.

        :param telegram_id: идентификатор пользователя в Telegram
        :param text: текст сообщения
        :return: текст ответа для отправки в Telegram
        """
        user_id = await self._get_or_create_user(telegram_id)

        # Onboarding: без сайтов просим назвать площадки и не запускаем диспетчер
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
            return self._build_reply(result)

        key = await self._router.classify(text)
        handler = self._handlers.get(key)
        if handler is None:
            return self._not_available_reply()
        return await handler(user_id, text, config)

    async def handle_document(
        self, telegram_id: str, filename: str, content: bytes
    ) -> str:
        """Обработать присланный документ как резюме и сохранить профиль + чанки.

        :param telegram_id: идентификатор пользователя в Telegram
        :param filename: исходное имя файла
        :param content: байты файла
        :return: текст ответа для отправки в Telegram
        """
        user_id = await self._get_or_create_user(telegram_id)

        suffix = Path(filename).suffix.lower()
        if suffix not in _RESUME_EXTENSIONS:
            return "Пришли резюме в формате PDF или DOCX."

        text = await self._read_cv_text(suffix, content)
        if not text.strip():
            return "Не удалось прочитать документ."

        # Структурированный профиль из текста CV
        profile = await extract_profile(self._gateway, text)
        if self._profile_repo is not None:
            await self._profile_repo.save(user_id, profile)

        # Чанки + эмбеддинги для RAG; при сбое — профиль уже сохранён
        try:
            await ingest_resume_chunks(
                self._session_factory, self._gateway.embedder, user_id, text
            )
        except Exception as exc:  # noqa: BLE001 - RAG не должен ронять загрузку CV
            logger.warning("RAG ingest failed for user %s: %s", user_id, exc)

        return (
            f"Записал профиль: роль — {profile.desired_role or '—'}, "
            f"навыков — {len(profile.skills)}."
        )

    async def _read_cv_text(self, suffix: str, content: bytes) -> str:
        """Извлечь текст из байтов документа через временный файл."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            return await extract_cv_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _handle_search(self, user_id: int, text: str, config: dict) -> str:
        """Способность search_job: запустить граф поиска и собрать ответ."""
        result = await self._graph.ainvoke(
            {"user_id": user_id, "user_message": text}, config
        )
        return self._build_reply(result)

    async def _handle_stats(self, user_id: int, text: str, config: dict) -> str:
        """Способность stats: вернуть накопленную статистику из БД."""
        if self._stats_provider is None:
            return "Статистика недоступна"
        stats = await self._stats_provider(user_id)
        return (
            f"Всего откликов: {stats.get('applied_count', 0)}, "
            f"общений с работодателями: {stats.get('replied_count', 0)}"
        )

    async def _handle_upload_resume(self, user_id: int, text: str, config: dict) -> str:
        """Способность upload_resume: попросить прислать файл."""
        return "Пришли резюме в формате PDF или DOCX сюда."

    async def _handle_chat(self, user_id: int, text: str, config: dict) -> str:
        """Способность chat: свободный ответ через текстовую модель."""
        reply_msg = await self._gateway.text_model.ainvoke(
            [
                (
                    "system",
                    "Ты — дружелюбный ассистент по поиску работы. "
                    "Отвечай кратко на русском.",
                ),
                ("human", text),
            ]
        )
        return reply_msg.content

    def _not_available_reply(self) -> str:
        """Ответ, когда ни одна способность не подошла."""
        abilities = ", ".join(c.description for c in self._capabilities)
        return f"Такого функционала пока нет. Я умею: {abilities}."

    async def _get_or_create_user(self, telegram_id: str) -> int:
        """Найти или создать пользователя по telegram_id."""
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
        """Определить, является ли сообщение подтверждением действия."""
        lower = text.lower().strip()
        if _has_keyword(lower, _REJECT_WORDS):
            return False
        return _has_keyword(lower, _APPROVE_WORDS)

    def _build_reply(self, result: dict) -> str:
        """Сформировать текст ответа из результата работы графа."""
        interrupts = result.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value
            job = pending.get("job", {})
            title = job.get("title", "вакансию")
            return f"Подтвердить отклик на «{title}»? Ответь «да» или «нет»."
        return result.get("reply", "Готово")
