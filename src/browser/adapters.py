class HhAdapter:
    """Адаптер для hh.ru: известные URL и CSS-селекторы."""

    BASE_URL = "https://hh.ru"
    SEARCH_URL = "https://hh.ru/search/vacancy"

    APPLY_BUTTON = (
        "a[data-qa='vacancy-response-link-top'], a[data-qa='vacancy-response-submit']"
    )
    CHAT_INPUT = "textarea[data-qa='messenger-input'], textarea[data-qa='chat_input']"
    CHAT_SEND = "button[data-qa='messenger-send'], button[data-qa='chat_send_button']"
    CHAT_MESSAGES = "[data-qa='messenger-message-text'], .chat-message"


class GenericAdapter:
    """Универсальный адаптер: агент сам разбирает страницу через extract()."""

    # Ключевые слова для поиска кнопки отклика на произвольном сайте
    APPLY_HINTS = ("откликнуться", "отклик", "apply", "отправить резюме", "send resume")
