import httpx


class BrowserExecutor:
    """HTTP-клиент к browser-сервису. Агент зависит только от этого интерфейса."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def navigate(self, user_id: int, url: str) -> dict:
        """Перейти на URL в браузере пользователя.

        :param user_id: идентификатор пользователя
        :param url: целевой URL
        :return: словарь с итоговым url и title
        """
        r = await self._client.post("/navigate", json={"user_id": user_id, "url": url})
        r.raise_for_status()
        return r.json()

    async def extract(self, user_id: int) -> dict:
        """Извлечь упрощённое представление текущей страницы.

        :param user_id: идентификатор пользователя
        :return: словарь с url, title, текстом и интерактивными элементами
        """
        r = await self._client.post("/extract", json={"user_id": user_id})
        r.raise_for_status()
        return r.json()

    async def screenshot(self, user_id: int) -> bytes:
        """Получить скриншот текущей страницы (PNG-байты).

        :param user_id: идентификатор пользователя
        :return: PNG-байты скриншота
        """
        r = await self._client.post("/screenshot", json={"user_id": user_id})
        r.raise_for_status()
        return r.content

    async def click(self, user_id: int, selector: str) -> dict:
        """Кликнуть по элементу на странице.

        :param user_id: идентификатор пользователя
        :param selector: CSS-селектор элемента
        :return: словарь с результатом операции
        """
        r = await self._client.post(
            "/click", json={"user_id": user_id, "selector": selector}
        )
        r.raise_for_status()
        return r.json()

    async def type_text(self, user_id: int, selector: str, text: str) -> dict:
        """Ввести текст в поле на странице.

        :param user_id: идентификатор пользователя
        :param selector: CSS-селектор поля ввода
        :param text: вводимый текст
        :return: словарь с результатом операции
        """
        r = await self._client.post(
            "/type", json={"user_id": user_id, "selector": selector, "text": text}
        )
        r.raise_for_status()
        return r.json()

    async def send_message(self, user_id: int, text: str) -> dict:
        """Отправить сообщение в чат на странице.

        :param user_id: идентификатор пользователя
        :param text: текст сообщения
        :return: словарь с результатом операции
        """
        r = await self._client.post("/message", json={"user_id": user_id, "text": text})
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        """Закрыть HTTP-клиент и освободить ресурсы."""
        await self._client.aclose()
