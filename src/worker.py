from arq.connections import RedisSettings

from config import settings


async def startup(ctx: dict) -> None:
    """Инициализация worker-контекста при старте."""


async def shutdown(ctx: dict) -> None:
    """Очистка ресурсов при остановке."""


async def ping(ctx: dict) -> str:
    """Проверочная задача: возвращает "pong"."""
    return "pong"


class WorkerSettings:
    """Настройки ARQ worker: функции и подключение к Redis."""

    functions = [ping]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
