from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Состояние агента, сериализуемое и передаваемое между узлами графа."""

    user_id: int
    user_message: str
    candidates: list[dict]  # найденные вакансии
    decisions: list[dict]  # решение по каждой вакансии (apply/skip + причина)
    pending_action: dict | None  # действие, ожидающее подтверждения человека
    needs_human: bool  # флаг: требуется вмешательство человека
    report: dict | None  # итоговая статистика для Telegram
    reply: str  # текст ответа пользователю
