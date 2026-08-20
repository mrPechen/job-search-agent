from typing import Any

from langchain_core.tools import BaseTool

# Разрешённые по умолчанию внешние инструменты (allowlist) — остальные отбрасываются.
# По умолчанию пусто: агент не получает ни одного внешнего инструмента, пока
# вызывающий код явно не разрешит конкретные имена.
DEFAULT_ALLOWED_TOOLS: set[str] = set()


def filter_tools(tools: list[BaseTool], allowed_names: set[str]) -> list[BaseTool]:
    """Оставить только инструменты из allowlist.

    Внешние MCP-серверы потенциально вредоносны, поэтому агент получает
    только явно разрешённые инструменты.

    :param tools: все инструменты, загруженные с MCP-серверов
    :param allowed_names: разрешённые имена инструментов
    :return: отфильтрованный список инструментов
    """
    return [t for t in tools if t.name in allowed_names]


class McpClient:
    """Клиент MCP: загружает инструменты внешних серверов в инструменты агента."""

    def __init__(
        self,
        servers: dict[str, Any],
        allowed_tools: set[str] | None = None,
    ) -> None:
        # Конфигурации подключений: имя сервера → параметры соединения
        # (stdio/http/websocket), в формате, ожидаемом MultiServerMCPClient.
        self._servers = servers
        # Allowlist по умолчанию пуст — безопасно «ничего не подключаем».
        self._allowed = (
            allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS
        )

    async def load_tools(self) -> list[BaseTool]:
        """Загрузить инструменты со всех настроенных серверов и применить allowlist.

        :return: список инструментов, прошедших фильтрацию по allowlist
        """
        # Импорт внутри метода: тяжёлая зависимость подключается только
        # когда действительно нужна загрузка внешних инструментов.
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # Клиент сам поднимает сессию на каждый вызов инструмента; контекстный
        # менеджер в версии 0.3.x удалён, поэтому используем прямой вызов.
        client = MultiServerMCPClient(self._servers)
        tools = await client.get_tools()
        return filter_tools(tools, self._allowed)
