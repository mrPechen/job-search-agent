from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


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
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=base_url)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def build_vision_model(
    provider: str, model: str, api_key: str, base_url: str = ""
) -> BaseChatModel:
    """Создать vision-модель (текст + изображения)."""
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=base_url)
    raise ValueError(f"Неизвестный провайдер: {provider}")
