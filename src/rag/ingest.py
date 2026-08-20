from pathlib import Path

from langchain_openai import OpenAIEmbeddings


async def extract_cv_text(path: Path) -> str:
    """Извлечь текст из CV (PDF или DOCX).

    :param path: путь к файлу резюме
    :return: сырой текст документа
    :raises ValueError: если формат не поддерживается
    """
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if path.suffix.lower() == ".docx":
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


def _split_long(text: str, max_chars: int) -> list[str]:
    """Разбить слишком длинный текст на части по границам слов."""
    parts: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def chunk_by_sections(text: str, max_chars: int = 1500) -> list[str]:
    """Разбить текст CV на чанки по пустым строкам (секциям)."""
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if not line.strip() and current:
            chunks.extend(
                _split_long(current, max_chars)
                if len(current) > max_chars
                else [current]
            )
            current = ""
        else:
            current += line + "\n"
    if current.strip():
        chunks.extend(
            _split_long(current, max_chars) if len(current) > max_chars else [current]
        )
    return [c.strip() for c in chunks if c.strip()]


def build_embeddings(
    provider: str, model: str, api_key: str, base_url: str = ""
) -> OpenAIEmbeddings:
    """Создать модель эмбеддингов. Провайдер openai (Ollama-эмбеддинги — позже)."""
    if provider == "openai":
        return OpenAIEmbeddings(model=model, api_key=api_key)
    raise ValueError(f"Провайдер эмбеддингов не поддерживается: {provider}")


async def embed_chunks(
    embedder: OpenAIEmbeddings, chunks: list[str]
) -> list[list[float]]:
    """Посчитать эмбеддинги для списка чанков."""
    return await embedder.aembed_documents(chunks)
