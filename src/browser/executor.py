import httpx

from config import settings


class BrowserExecutor:
    """HTTP-клиент к browser-сервису. Агент зависит только от этого интерфейса."""

    def __init__(
        self, base_url: str, timeout: float = 60.0, api_token: str | None = None
    ) -> None:
        token = settings.BROWSER_API_TOKEN if api_token is None else api_token
        headers = {"x-browser-token": token} if token else None
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=headers
        )

    async def navigate(
        self, user_id: int, url: str, allowed_domains: list[str] | None = None
    ) -> dict:
        """Перейти на URL в браузере пользователя.

        :param user_id: идентификатор пользователя
        :param url: целевой URL
        :param allowed_domains: список разрешённых доменов для проверки на сервере
        :return: словарь с итоговым url и title
        """
        payload: dict = {"user_id": user_id, "url": url}
        if allowed_domains is not None:
            payload["allowed_domains"] = allowed_domains
        r = await self._client.post("/navigate", json=payload)
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

    async def scroll(self, user_id: int, delta: int = 800) -> dict:
        """Прокрутить страницу на delta пикселей вниз."""
        r = await self._client.post(
            "/scroll", json={"user_id": user_id, "delta": delta}
        )
        r.raise_for_status()
        return r.json()

    async def back(self, user_id: int) -> dict:
        """Вернуться на предыдущую страницу."""
        r = await self._client.post("/back", json={"user_id": user_id})
        r.raise_for_status()
        return r.json()

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
