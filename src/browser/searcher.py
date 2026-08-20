from src.browser.adapters import HhAdapter


class HhSearcher:
    """Поиск вакансий на hh.ru через browser-executor.

    Внимание: требует авторизованной сессии (персистентный профиль Playwright).
    Для MVP использует упрощённый парсинг результатов поиска из extract().
    """

    def __init__(self, browser, default_query: str = "python developer") -> None:
        self._browser = browser
        self._default_query = default_query

    async def __call__(self, user_id: int, query: str = "") -> list[dict]:
        """Найти вакансии по запросу и вернуть список кандидатов.

        :param user_id: идентификатор пользователя
        :param query: поисковый запрос (сообщение пользователя); пустой → default_query
        :return: список вакансий в виде словарей (title + url)
        """
        text = (query or self._default_query).replace(" ", "+")
        url = f"{HhAdapter.SEARCH_URL}?text={text}"
        await self._browser.navigate(user_id, url)
        page = await self._browser.extract(user_id)
        return self._parse_candidates(page)

    def _parse_candidates(self, page: dict) -> list[dict]:
        """Извлечь вакансии из упрощённого представления страницы.

        :param page: словарь с элементами страницы (результат extract())
        :return: список вакансий (title + url)
        """
        candidates = []
        for el in page.get("elements", []):
            # Ищем ссылки на вакансии по тексту/атрибутам
            text = (el.get("text") or "").strip()
            if not text or el.get("tag") != "a":
                continue
            candidates.append({"title": text, "url": el.get("selector", "")})
        return candidates
