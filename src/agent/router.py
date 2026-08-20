import re
from typing import Literal

from pydantic import BaseModel


class Intent(BaseModel):
    """Классифицированное намерение пользователя."""

    intent: Literal["search_job", "stats", "confirm", "chat"]


# Ключевые слова для определения намерения без обращения к LLM (fallback и тесты)
_SEARCH_WORDS = ("работ", "ваканс", "ищ", "поиск", "найти", "поработать", "собесед")
_STATS_WORDS = ("стат", "отклик", "сколько", "результат", "итог")
_CONFIRM_WORDS = ("да", "ок", "подтвержд", "соглас", "отправ", "давай", "го")


def _has_keyword(lower: str, words: tuple[str, ...]) -> bool:
    """Проверить вхождение любого ключевого слова по границе слова.

    Граница слова нужна, чтобы короткие слова («го», «да», «ок») не совпадали
    внутри обычных слов (например, «го» внутри «нового»).
    """
    return any(re.search(rf"\b{re.escape(word)}", lower) for word in words)


def keyword_classify(message: str) -> Intent:
    """Классифицировать намерение по ключевым словам (без LLM).

    Используется как быстрый fallback и в тестах. Приоритет: confirm > stats > search.
    """
    lower = message.lower()
    if _has_keyword(lower, _CONFIRM_WORDS):
        return Intent(intent="confirm")
    if _has_keyword(lower, _STATS_WORDS):
        return Intent(intent="stats")
    if _has_keyword(lower, _SEARCH_WORDS):
        return Intent(intent="search_job")
    return Intent(intent="chat")


class IntentRouter:
    """Классификация намерения пользователя с приоритетом LLM и keyword-fallback."""

    def __init__(self, gateway) -> None:
        self._gateway = gateway

    async def classify(self, message: str) -> Intent:
        """Классифицировать сообщение. При недоступности/ошибке LLM — keyword fallback."""
        try:
            return await self._gateway.invoke_structured(
                self._gateway.text_model,
                [
                    (
                        "system",
                        "Классифицируй намерение пользователя поиска работы: "
                        "search_job (искать/откликаться на вакансии), stats (статистика), "
                        "confirm (подтверждение действия), chat (остальное).",
                    ),
                    ("human", message),
                ],
                Intent,
            )
        except Exception:
            # LLM недоступен — используем детерминированный fallback по ключевым словам
            # Широкий except намеренен: роутер не должен ронять бота при любой ошибке LLM
            return keyword_classify(message)
