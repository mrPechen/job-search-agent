import re

from pydantic import BaseModel


class Capability(BaseModel):
    """Одна способность агента: ключ + описание для LLM-промпта."""

    key: str
    description: str


class CapabilityChoice(BaseModel):
    """Результат выбора способности. key=None — ни одна не подходит."""

    key: str | None = None


# Ключевые слова для детерминированного fallback (по границе слова)
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "upload_resume": ("резюме", "cv"),
    "stats": ("стат", "сколько", "итог", "результат"),
    "search_job": ("работ", "ваканс", "ищ", "поиск", "найти", "собесед"),
}


def _has_keyword(lower: str, words: tuple[str, ...]) -> bool:
    """Проверить вхождение любого ключевого слова по границе слова."""
    return any(re.search(rf"\b{re.escape(word)}", lower) for word in words)


class CapabilityRouter:
    """Сопоставление сообщения со способностью агента (LLM + keyword-fallback)."""

    def __init__(self, gateway, capabilities: list[Capability]) -> None:
        self._gateway = gateway
        self._capabilities = capabilities

    @property
    def capabilities(self) -> list[Capability]:
        """Список доступных способностей (для перечня «я умею»)."""
        return self._capabilities

    async def classify(self, message: str) -> str | None:
        """Вернуть ключ способности или None, если ни одна не подходит."""
        try:
            choice = await self._gateway.invoke_structured(
                self._gateway.text_model,
                [
                    ("system", self._build_prompt()),
                    ("human", message),
                ],
                CapabilityChoice,
            )
            if choice.key in {c.key for c in self._capabilities}:
                return choice.key
            return None
        except Exception:
            # LLM недоступен — детерминированный fallback по ключевым словам
            return self._keyword_classify(message)

    def _build_prompt(self) -> str:
        lines = "\n".join(
            f"- {c.key} — {c.description}" for c in self._capabilities
        )
        return (
            "Ты — маршрутизатор личного агента поиска работы. "
            "Доступные возможности:\n"
            f"{lines}\n"
            "Верни ключ одной из возможностей. Если это обычный разговор или "
            "вопрос — верни chat. Если пользователь просит выполнить действие, "
            'которого нет в списке, — верни "none".'
        )

    def _keyword_classify(self, message: str) -> str:
        lower = message.lower()
        for cap in self._capabilities:
            if cap.key == "chat":
                continue
            if _has_keyword(lower, _KEYWORDS.get(cap.key, ())):
                return cap.key
        return "chat"
