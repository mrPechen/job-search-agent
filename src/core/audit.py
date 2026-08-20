import logging

logger = logging.getLogger("audit")


def log_action(user_id: int, action: str, decision: str, details: dict) -> None:
    """Записать действие агента в журнал аудита.

    Фиксирует цепочку «что агент хотел сделать → какое решение принято»,
    чтобы можно было разобрать инциденты безопасности постфактум.

    :param user_id: внутренний id пользователя
    :param action: имя действия (apply, send_message, ...)
    :param decision: решение (approved | rejected | executed)
    :param details: детали действия (вакансия, текст и т.п.)
    """
    logger.info(
        "AUDIT user_id=%s action=%s decision=%s details=%s",
        user_id,
        action,
        decision,
        details,
    )
