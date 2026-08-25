import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class Sites(BaseModel):
    """Список доменов, распознанных из сообщения пользователя."""

    domains: list[str] = Field(default_factory=list)


_DOMAIN_RE = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:[a-z]{2,})", re.IGNORECASE
)


def normalize_domain(raw: str) -> str | None:
    """Нормализовать домен: убрать схему/путь/порт/www, привести к нижнему регистру."""
    value = raw.strip().lower()
    if not value:
        return None
    if "://" in value or value.startswith("www."):
        parsed = urlparse(value if "://" in value else f"//{value}")
        host = parsed.hostname
    else:
        host = value.split("/")[0].split(":")[0]
    if not host:
        return None
    host = host.removeprefix("www.")
    if "." not in host:
        return None
    return host


def extract_domains(text: str) -> list[str]:
    """Извлечь уникальные домены из текста (regex-фолбэк)."""
    result: list[str] = []
    for match in _DOMAIN_RE.findall(text):
        domain = normalize_domain(match)
        if domain and domain not in result:
            result.append(domain)
    return result


async def parse_sites_message(gateway, text: str) -> list[str]:
    """Распарсить сообщение в список доменов: сначала LLM, затем regex-фолбэк."""
    try:
        sites = await gateway.invoke_structured(
            gateway.text_model,
            [
                (
                    "system",
                    "Извлеки домены сайтов по поиску работы из сообщения. "
                    "Верни только hostname (без схемы, пути и www).",
                ),
                ("human", text),
            ],
            Sites,
        )
        domains = [d for d in (normalize_domain(x) for x in sites.domains) if d]
        if domains:
            return list(dict.fromkeys(domains))
    except Exception:
        pass
    return extract_domains(text)


class SitesRepository:
    """Доступ к сохранённым сайтам пользователя."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def get_domains(self, user_id: int) -> list[str]:
        from sqlalchemy import select

        from src.database.models import UserSite

        async with self._session_factory() as session:
            result = await session.execute(
                select(UserSite.domain)
                .where(UserSite.user_id == user_id)
                .order_by(UserSite.id)
            )
            return list(result.scalars().all())

    async def add_domains(self, user_id: int, domains: list[str]) -> None:
        from src.database.models import UserSite

        existing = set(await self.get_domains(user_id))
        async with self._session_factory() as session:
            for domain in dict.fromkeys(domains):
                if domain in existing:
                    continue
                session.add(UserSite(user_id=user_id, domain=domain))
                existing.add(domain)
            await session.commit()
