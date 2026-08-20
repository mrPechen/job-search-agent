import re

# Паттерны, указывающие на возможную утечку секретов в исходящем сообщении
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),  # OpenAI-подобные ключи
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{10,}"),  # токены авторизации
    re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),  # приватные ключи
]

MAX_MESSAGE_LEN = 4000


def check_outgoing_message(text: str) -> tuple[bool, str]:
    """Проверить исходящее сообщение перед отправкой работодателю.

    :param text: текст сообщения
    :return: (ok, reason) — ok=True если сообщение безопасно, иначе причина блокировки
    """
    if not text or not text.strip():
        return False, "сообщение пустое"
    if len(text) > MAX_MESSAGE_LEN:
        return False, f"сообщение слишком длинное ({len(text)} > {MAX_MESSAGE_LEN})"
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return (
                False,
                "сообщение содержит подозрительные данные (возможна утечка секретов)",
            )
    return True, ""
