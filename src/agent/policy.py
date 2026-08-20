# Уровни риска: чем выше, тем опаснее действие для пользователя
RISK_LEVELS = {"read": 0, "draft": 1, "high_risk": 2}

# Отображение действия → уровень риска (единая точка принятия решения о безопасности)
_ACTION_RISK = {
    "search": "read",
    "extract": "read",
    "read_chat": "read",
    "screenshot": "read",
    "draft_cover_letter": "draft",
    "draft_reply": "draft",
    "apply": "high_risk",
    "send_message": "high_risk",
}


def classify_action(action: str) -> str:
    """Вернуть уровень риска действия.

    :param action: имя действия
    :return: read | draft | high_risk (неизвестные действия считаются read)
    """
    return _ACTION_RISK.get(action, "read")


def requires_human_approval(action: str) -> bool:
    """Требует ли действие подтверждения человека перед выполнением."""
    return classify_action(action) == "high_risk"
