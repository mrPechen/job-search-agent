from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from config import settings
from src.database.db_settings import Base


class User(Base):
    """Пользователь телеграм-бота с настройками и статусом."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="active")


class UserSite(Base):
    """Сайт по поиску работы, добавленный пользователем."""

    __tablename__ = "user_sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    domain: Mapped[str] = mapped_column(String(255))
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("user_id", "domain", name="uq_user_sites_user_domain"),
    )


class Profile(Base):
    """Профиль соискателя: навыки, опыт и пожелания по вакансии."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    skills: Mapped[list] = mapped_column(JSONB, default=list)
    experience: Mapped[list] = mapped_column(JSONB, default=list)
    desired_role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    desired_salary: Mapped[str | None] = mapped_column(String(128), nullable=True)
    desired_location: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Document(Base):
    """Документ (резюме) с разбиением на чанки и эмбеддингами для поиска."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)


class JobSite(Base):
    """Площадка для поиска вакансий с ключом адаптера и учётными данными."""

    __tablename__ = "job_sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    adapter_key: Mapped[str] = mapped_column(String(64))
    credentials: Mapped[str] = mapped_column(Text, default="")


class Job(Base):
    """Вакансия, найденная на площадке, со статусом обработки."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("job_sites.id"))
    external_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(512))
    company: Mapped[str | None] = mapped_column(String(512), nullable=True)
    url: Mapped[str] = mapped_column(String(1024))
    description: Mapped[str] = mapped_column(Text, default="")
    raw: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="new")


class Application(Base):
    """Отклик пользователя на вакансию со статусом и сопроводительным письмом."""

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    cover_letter: Mapped[str] = mapped_column(Text, default="")
    applied_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Conversation(Base):
    """Переписка с работодателем по отклику или напрямую на площадке."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id"), nullable=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    site_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_sites.id"), nullable=True
    )
    external_thread_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), default="employer")
    status: Mapped[str] = mapped_column(String(32), default="active")


class Message(Base):
    """Сообщение в переписке с признаком необходимости ручной проверки."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="agent")
    needs_review: Mapped[bool] = mapped_column(default=False)
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SearchRun(Base):
    """Запуск поиска вакансий со счётчиками откликов и ответов."""

    __tablename__ = "search_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    trigger: Mapped[str] = mapped_column(String(64))
    filters: Mapped[dict] = mapped_column(JSONB, default=dict)
    applied_count: Mapped[int] = mapped_column(default=0)
    replied_count: Mapped[int] = mapped_column(default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Notification(Base):
    """Уведомление пользователю в телеграм со статусом отправки."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    sent: Mapped[bool] = mapped_column(default=False)


class LlmCall(Base):
    """Запись о вызове LLM: модель, токены, стоимость и задержка."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    tokens: Mapped[int] = mapped_column(default=0)
    cost: Mapped[float] = mapped_column(default=0.0)
    latency_ms: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
