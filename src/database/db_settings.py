from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool

from config import settings


class Base(DeclarativeBase):
    """База для всех ORM-моделей."""


sql_link = (
    f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASS}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

engine = create_async_engine(sql_link, echo=False, poolclass=NullPool)
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, class_=AsyncSession, bind=engine
)
