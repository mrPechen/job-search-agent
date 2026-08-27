import logging

from config import settings
from src.agent.browser_loop import ApplyOutcome, BrowserLoop, SearchOutcome

logger = logging.getLogger(__name__)

# Максимальная длина текста вакансии, подаваемого в генерацию сопроводительного
_VACANCY_TEXT_LIMIT = 12000


class UniversalSearcher:
    """Поиск вакансий на всех сайтах пользователя через VLM-цикл."""

    def __init__(
        self, executor, gateway, sites, profile_provider=None, loop=None
    ) -> None:
        self._executor = executor
        self._loop = loop if loop is not None else BrowserLoop(executor, gateway)
        self._sites = sites
        self._profile_provider = profile_provider

    async def read_vacancy(self, user_id: int, url: str) -> str:
        """Открыть вакансию и вернуть её полный текст (для сопроводительного).

        Используется перед генерацией письма, чтобы учесть требования,
        которые работодатель указывает в описании вакансии.

        :param user_id: внутренний id пользователя
        :param url: адрес страницы вакансии
        :return: полный текст страницы (обрезанный до лимита) или ""
        """
        if not url:
            return ""
        domains = await self._sites.get_domains(user_id)
        await self._executor.navigate(user_id, url, allowed_domains=domains)
        page = await self._executor.extract(user_id)
        return (page.get("text") or "").strip()[:_VACANCY_TEXT_LIMIT]

    async def __call__(self, user_id: int, query: str = "") -> list[dict]:
        domains = await self._sites.get_domains(user_id)
        search_query = await self._resolve_query(user_id, query) or "вакансии"
        candidates: list[dict] = []
        for domain in domains:
            start_url, goal = self._plan_domain(domain, search_query)
            try:
                outcome: SearchOutcome = await self._loop.run(
                    user_id,
                    goal,
                    SearchOutcome,
                    allowed_domains=domains,
                    start_url=start_url,
                )
            except Exception as exc:
                logger.warning("Search on %s failed: %s", domain, exc)
                continue
            if outcome.error:
                logger.warning("Search on %s failed: %s", domain, outcome.error)
            for candidate in outcome.candidates:
                candidate.setdefault("site", domain)
            candidates.extend(outcome.candidates)
        return candidates[: settings.MAX_CANDIDATES]

    def _plan_domain(self, domain: str, query: str) -> tuple[str, str]:
        """Вернуть (start_url, goal) для поиска на домене.

        Для hh.ru при заданном HH_SEARCH_URL сразу открываем готовую выдачу
        с фильтрами, минуя ручной ввод запроса.
        """
        if settings.HH_SEARCH_URL and "hh.ru" in domain:
            goal = (
                "Перед тобой страница с результатами поиска вакансий. "
                "Собери для каждой вакансии: title, url, короткое описание. "
                "url каждой вакансии бери из атрибута href её ссылки "
                "(поле href в списке элементов)."
            )
            return settings.HH_SEARCH_URL, goal
        goal = (
            f"Найди на сайте {domain} поле поиска вакансий, введи запрос "
            f"«{query}», получи список вакансий со страницы результатов "
            "и собери для каждой: title, url, короткое описание. "
            "url каждой вакансии бери из атрибута href её ссылки "
            "(поле href в списке элементов)."
        )
        return f"https://{domain}", goal

    async def _resolve_query(self, user_id: int, query: str) -> str:
        if self._profile_provider is not None:
            profile = await self._profile_provider(user_id)
            role = (profile or {}).get("desired_role")
            if role:
                return role
        return query


class UniversalApplier:
    """Отклик на вакансию через VLM-цикл."""

    def __init__(self, executor, gateway, sites, loop=None) -> None:
        self._loop = loop if loop is not None else BrowserLoop(executor, gateway)
        self._sites = sites

    async def __call__(self, user_id: int, decision: dict) -> ApplyOutcome:
        job = decision.get("job") or {}
        url = job.get("url", "")
        cover = decision.get("cover_letter", "")
        resume = decision.get("resume", "")
        if not url:
            return ApplyOutcome(applied=False, error="missing url")
        domains = await self._sites.get_domains(user_id)
        parts = [
            f"Открой вакансию {url}, найди кнопку отклика "
            "(Откликнуться/Apply) и нажми её."
        ]
        if resume:
            parts.append(
                f"В форме отклика выбери резюме «{resume}» "
                "(если есть выбор/селектор резюме)."
            )
        parts.append(
            f"Впиши сопроводительное письмо: «{cover}», отправь и подтверди успех."
        )
        goal = " ".join(parts)
        try:
            return await self._loop.run(
                user_id, goal, ApplyOutcome, allowed_domains=domains, start_url=url
            )
        except Exception as exc:
            logger.warning("Apply failed: %s", exc)
            return ApplyOutcome(applied=False, error=str(exc))
