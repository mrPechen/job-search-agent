from functools import cached_property
from typing import Any

from langchain_core.language_models import BaseChatModel

from config import settings
from src.llm.providers import build_text_model, build_vision_model


class LLMGateway:
    """Единая точка доступа к текстовым и vision-моделям.

    Инкапсулирует выбор провайдера и предоставляет агентам единый
    интерфейс для генерации текста и структурированных ответов.

    Модели создаются лениво (при первом обращении через cached_property),
    чтобы конструктор не требовал ключа API и не выполнял сетевых вызовов.
    """

    @cached_property
    def text_model(self) -> BaseChatModel:
        """Текстовая модель, выбранная по настройкам провайдера."""
        return build_text_model(
            settings.LLM_PROVIDER,
            settings.LLM_TEXT_MODEL,
            settings.OPENAI_API_KEY,
            settings.OLLAMA_BASE_URL,
        )

    @cached_property
    def vision_model(self) -> BaseChatModel:
        """Vision-модель для работы с текстом и изображениями."""
        return build_vision_model(
            settings.LLM_PROVIDER,
            settings.LLM_VISION_MODEL,
            settings.OPENAI_API_KEY,
            settings.OLLAMA_BASE_URL,
        )

    @cached_property
    def embedder(self):
        """Модель эмбеддингов для RAG (провайдер совпадает с LLM_PROVIDER)."""
        from src.rag.ingest import build_embeddings

        return build_embeddings(
            settings.LLM_PROVIDER,
            settings.EMBEDDING_MODEL,
            settings.OPENAI_API_KEY,
            settings.OLLAMA_BASE_URL,
        )

    async def invoke_structured(
        self, model: BaseChatModel, messages: list, schema: type
    ) -> Any:
        """Вызвать модель с Pydantic-валидацией вывода.

        :param model: целевая модель
        :param messages: список сообщений для модели
        :param schema: Pydantic-класс для структурированного вывода
        :return: экземпляр schema, валидированный Pydantic
        """
        structured = model.with_structured_output(schema)
        return await structured.ainvoke(messages)
