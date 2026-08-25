import logging

from src.agent.browser_loop import ApplyOutcome, BrowserLoop, SearchOutcome

logger = logging.getLogger(__name__)


class UniversalSearcher:
    """Поиск вакансий на всех сайтах пользователя через VLM-цикл."""

    def __init__(
        self, executor, gateway, sites, profile_provider=None, loop=None
    ) -> None:
        self._loop = loop if loop is not None else BrowserLoop(executor, gateway)
        self._sites = sites
        self._profile_provider = profile_provider

    async def __call__(self, user_id: int, query: str = "") -> list[dict]:
        domains = await self._sites.get_domains(user_id)
        search_query = await self._resolve_query(user_id, query) or "вакансии"
        candidates: list[dict] = []
        for domain in domains:
            goal = (
                f"Найди на сайте {domain} поле поиска вакансий, введи запрос "
                f"«{search_query}», получи список вакансий со страницы результатов "
                "и собери для каждой: title, url, короткое описание."
            )
            outcome: SearchOutcome = await self._loop.run(
                user_id,
                goal,
                SearchOutcome,
                allowed_domains=domains,
                start_url=f"https://{domain}",
            )
            if outcome.error:
                logger.warning("Search on %s failed: %s", domain, outcome.error)
            for candidate in outcome.candidates:
                candidate.setdefault("site", domain)
            candidates.extend(outcome.candidates)
        return candidates

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
        if not url:
            return ApplyOutcome(applied=False, error="missing url")
        domains = await self._sites.get_domains(user_id)
        goal = (
            f"Открой вакансию {url}, найди кнопку отклика (Откликнуться/Apply), "
            f"нажми её, впиши сопроводительное письмо: «{cover}», отправь и "
            "подтверди успех."
        )
        return await self._loop.run(
            user_id, goal, ApplyOutcome, allowed_domains=domains, start_url=url
        )
