from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings


def _build_ollama(model: str, base_url: str, num_ctx: int) -> BaseChatModel:
    """Создать ChatOllama с заданным контекстом и постоянной загрузкой модели.

    keep_alive=-1 держит модель в памяти между запросами, чтобы не платить
    повторной загрузкой весов на каждом шаге цикла браузера.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        num_ctx=num_ctx,
        keep_alive=-1,
    )


def build_text_model(
    provider: str, model: str, api_key: str, base_url: str = ""
) -> BaseChatModel:
    """Создать текстовую модель по имени провайдера.

    :param provider: имя провайдера (openai | ollama)
    :param model: имя модели
    :param api_key: ключ API (для openai)
    :param base_url: базовый URL (для ollama)
    :return: готовая к вызову языковая модель
    :raises ValueError: если провайдер неизвестен
    """
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=settings.OPENAI_BASE_URL or None,
            extra_body=settings.LLM_EXTRA_BODY or None,
            temperature=0.2,
        )
    if provider == "ollama":
        return _build_ollama(model, base_url, settings.OLLAMA_NUM_CTX)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def build_vision_model(
    provider: str, model: str, api_key: str, base_url: str = ""
) -> BaseChatModel:
    """Создать vision-модель (текст + изображения)."""
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=settings.OPENAI_BASE_URL or None,
            temperature=0.2,
        )
    if provider == "ollama":
        return _build_ollama(model, base_url, settings.OLLAMA_VISION_NUM_CTX)
    raise ValueError(f"Неизвестный провайдер: {provider}")
