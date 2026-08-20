from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Жизненный цикл приложения: здесь создаются и закрываются клиенты."""
    yield


app = FastAPI(title="Job Search Agent", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Проверка живости сервиса."""
    return {"ok": True}
