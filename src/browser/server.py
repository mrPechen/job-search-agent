import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Response
from playwright.async_api import BrowserContext, Page, Playwright, async_playwright
from pydantic import BaseModel

from src.browser.adapters import HhAdapter

logger = logging.getLogger(__name__)

# Домены, на которые разрешено ходить браузеру (граница области действия агента)
ALLOWED_DOMAINS = {"hh.ru", "www.hh.ru", "localhost", "127.0.0.1"}

# Минимальный интервал между действиями одного пользователя (анти-бан)
MIN_ACTION_INTERVAL = 1.5

# JS для извлечения видимого текста страницы
_BODY_TEXT_JS = "() => (document.body ? document.body.innerText : '')"

# JS для перечисления интерактивных элементов с построением CSS-селектора
_EXTRACT_JS = """
() => {
  const buildSelector = (el) => {
    if (el.id) return "#" + el.id;
    const qa = el.getAttribute("data-qa");
    if (qa) return '[data-qa="' + qa + '"]';
    if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
    const cls = typeof el.className === "string"
      ? el.className.trim().split(" ").filter(Boolean)[0] || ""
      : "";
    if (cls) return el.tagName.toLowerCase() + "." + cls;
    return el.tagName.toLowerCase();
  };
  const els = Array.from(document.querySelectorAll("a, button, input, textarea, select"));
  return els.map((el) => ({
    tag: el.tagName.toLowerCase(),
    text: (el.innerText || el.value || el.getAttribute("aria-label") || "").trim(),
    selector: buildSelector(el),
    type: el.getAttribute("type") || "",
  }));
}
"""


def _is_allowed_url(url: str) -> bool:
    """Проверить, что домен URL входит в whitelist (или является его поддоменом).

    :param url: целевой URL
    :return: True, если навигация разрешена
    """
    hostname = urlparse(url).hostname or ""
    return any(hostname == d or hostname.endswith("." + d) for d in ALLOWED_DOMAINS)


class NavigateRequest(BaseModel):
    """Тело запроса навигации на URL."""

    user_id: int
    url: str


class UserRequest(BaseModel):
    """Тело запроса с идентификатором пользователя."""

    user_id: int


class ClickRequest(BaseModel):
    """Тело запроса клика по элементу."""

    user_id: int
    selector: str


class TypeRequest(BaseModel):
    """Тело запроса ввода текста в элемент."""

    user_id: int
    selector: str
    text: str


class MessageRequest(BaseModel):
    """Тело запроса отправки сообщения в чат."""

    user_id: int
    text: str


class BrowserManager:
    """Управление пулом персистентных Playwright-контекстов по пользователям.

    Для каждого user_id создаётся отдельный профиль (user-data-dir), чтобы
    сохранять сессии/куки между запусками. Контексты создаются лениво и
    кэшируются в словаре. Каждое действие ограничено минимальным интервалом,
    чтобы вести себя «вежливо» и не попасть под анти-бан.
    """

    def __init__(
        self, profiles_dir: Path, min_interval: float = MIN_ACTION_INTERVAL
    ) -> None:
        self._profiles_dir = profiles_dir
        self._min_interval = min_interval
        self._contexts: dict[int, BrowserContext] = {}
        self._playwright: Playwright | None = None
        self._last_action: dict[int, float] = {}

    async def _get_playwright(self) -> Playwright:
        """Лениво запустить Playwright-драйвер."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        return self._playwright

    async def get_context(self, user_id: int) -> BrowserContext:
        """Вернуть (создав при необходимости) персистентный контекст пользователя.

        :param user_id: идентификатор пользователя
        :return: Playwright-контекст с отдельным профилем
        """
        if user_id not in self._contexts:
            playwright = await self._get_playwright()
            user_data_dir = self._profiles_dir / str(user_id)
            user_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Создаю браузерный профиль для user_id=%s", user_id)
            self._contexts[user_id] = (
                await playwright.chromium.launch_persistent_context(
                    user_data_dir=str(user_data_dir),
                    headless=True,
                )
            )
        return self._contexts[user_id]

    async def get_page(self, user_id: int) -> Page:
        """Вернуть текущую страницу пользователя (последнюю открытую)."""
        context = await self.get_context(user_id)
        if context.pages:
            return context.pages[-1]
        return await context.new_page()

    async def _throttle(self, user_id: int) -> None:
        """Выдержать минимальную паузу между действиями пользователя."""
        now = time.monotonic()
        last = self._last_action.get(user_id)
        if last is not None:
            wait = self._min_interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_action[user_id] = time.monotonic()

    async def navigate(self, user_id: int, url: str) -> dict:
        """Перейти на URL и вернуть итоговый адрес и заголовок страницы."""
        await self._throttle(user_id)
        page = await self.get_page(user_id)
        await page.goto(url, wait_until="domcontentloaded")
        return {"url": page.url, "title": await page.title()}

    async def extract(self, user_id: int) -> dict:
        """Вернуть упрощённое представление текущей страницы."""
        page = await self.get_page(user_id)
        text = await page.evaluate(_BODY_TEXT_JS)
        elements = await page.evaluate(_EXTRACT_JS)
        return {
            "url": page.url,
            "title": await page.title(),
            "text": text,
            "elements": elements,
        }

    async def screenshot(self, user_id: int) -> bytes:
        """Сделать скриншот текущей страницы (PNG)."""
        page = await self.get_page(user_id)
        return await page.screenshot()

    async def click(self, user_id: int, selector: str) -> dict:
        """Кликнуть по элементу; вернуть ошибку, если селектор не найден.

        :raises LookupError: если селектор отсутствует на странице
        """
        await self._throttle(user_id)
        page = await self.get_page(user_id)
        if not await page.query_selector(selector):
            raise LookupError(f"Селектор не найден: {selector}")
        await page.click(selector)
        return {"ok": True}

    async def type_text(self, user_id: int, selector: str, text: str) -> dict:
        """Заполнить поле ввода текстом.

        :raises LookupError: если селектор отсутствует на странице
        """
        await self._throttle(user_id)
        page = await self.get_page(user_id)
        if not await page.query_selector(selector):
            raise LookupError(f"Селектор не найден: {selector}")
        await page.fill(selector, text)
        return {"ok": True}

    async def send_message(self, user_id: int, text: str) -> dict:
        """Найти поле чата, ввести сообщение и отправить его.

        :raises LookupError: если на странице нет поля ввода сообщения
        """
        await self._throttle(user_id)
        page = await self.get_page(user_id)

        # Поиск поля ввода: сначала селекторы hh.ru, затем универсальные
        input_selector: str | None = None
        for candidate in (HhAdapter.CHAT_INPUT, "textarea", "[contenteditable]"):
            if await page.query_selector(candidate):
                input_selector = candidate
                break
        if input_selector is None:
            raise LookupError("Поле ввода сообщения не найдено")

        await page.fill(input_selector, text)

        # Отправка: кнопка «Отправить», иначе Enter в поле ввода
        if await page.query_selector(HhAdapter.CHAT_SEND):
            await page.click(HhAdapter.CHAT_SEND)
        else:
            await page.press(input_selector, "Enter")
        return {"ok": True}

    async def close(self) -> None:
        """Закрыть все контексты и остановить Playwright-драйвер."""
        for context in self._contexts.values():
            await context.close()
        self._contexts.clear()
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


def build_browser_app(profiles_dir: Path | None = None) -> FastAPI:
    """Собрать FastAPI-приложение browser-сервиса.

    :param profiles_dir: каталог для персистентных профилей (для тестов —
        временный каталог)
    :return: готовое FastAPI-приложение
    """
    manager = BrowserManager(
        profiles_dir=profiles_dir or (Path.cwd() / ".browser_profiles")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Контексты создаются лениво, поэтому на старте ничего запускать не нужно
        yield
        # Освобождаем браузерные профили при остановке сервиса
        await manager.close()
        logger.info("BrowserManager остановлен")

    app = FastAPI(title="Browser Executor", lifespan=lifespan)
    app.state.browser_manager = manager

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    @app.post("/navigate")
    async def navigate(req: NavigateRequest) -> dict:
        if not _is_allowed_url(req.url):
            raise HTTPException(status_code=403, detail=f"Домен не разрешён: {req.url}")
        return await manager.navigate(req.user_id, req.url)

    @app.post("/extract")
    async def extract(req: UserRequest) -> dict:
        return await manager.extract(req.user_id)

    @app.post("/screenshot")
    async def screenshot(req: UserRequest) -> Response:
        data = await manager.screenshot(req.user_id)
        return Response(content=data, media_type="image/png")

    @app.post("/click")
    async def click(req: ClickRequest) -> dict:
        try:
            return await manager.click(req.user_id, req.selector)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/type")
    async def type_text(req: TypeRequest) -> dict:
        try:
            return await manager.type_text(req.user_id, req.selector, req.text)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/message")
    async def send_message(req: MessageRequest) -> dict:
        try:
            return await manager.send_message(req.user_id, req.text)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    return app
