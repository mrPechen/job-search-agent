from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Document
from src.rag.schemas import ProfileData


async def store_chunks(
    session: AsyncSession,
    user_id: int,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """Сохранить чанки с эмбеддингами в БД. Возвращает число сохранённых чанков."""
    docs = [
        Document(user_id=user_id, chunk_text=chunk, embedding=emb)
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]
    session.add_all(docs)
    await session.flush()
    return len(docs)


async def retrieve_relevant(
    session: AsyncSession,
    user_id: int,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[Document]:
    """Найти наиболее релевантные чанки по косинусной близости (pgvector)."""
    query = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(query)
    return list(result.scalars().all())


async def extract_profile(gateway, cv_text: str) -> ProfileData:
    """Извлечь структурированный профиль из текста CV через LLM."""
    messages = [
        (
            "system",
            "Ты извлекаешь структуру из резюме. "
            "Верни JSON-объект с полями skills (список навыков), experience "
            "(список мест работы/достижений), desired_role, desired_salary, desired_location.",
        ),
        ("human", cv_text),
    ]
    return await gateway.invoke_structured(gateway.text_model, messages, ProfileData)
