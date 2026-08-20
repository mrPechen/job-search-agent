import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from src.database import models  # noqa: F401  # регистрируем модели в metadata
from src.database.db_settings import Base, engine, sql_link

# Объект конфигурации Alembic с доступом к значениям из alembic.ini
config = context.config

# Настройка логгеров из alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Берём URL асинхронного движка из db_settings, чтобы не дублировать параметры БД
config.set_main_option("sqlalchemy.url", sql_link)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Запуск миграций в оффлайн-режиме: только SQL-скрипт без подключения к БД."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Оборачивает синхронное подключение в контекст миграции."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Запуск миграций через async-движок: run_sync пробрасывает синхронное соединение."""
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)


def run_migrations_online() -> None:
    """Точка входа онлайн-режима: asyncio.run для асинхронных миграций."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
