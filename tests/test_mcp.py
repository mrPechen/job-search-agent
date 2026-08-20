from langchain_core.tools import tool

from src.mcp.client import filter_tools


@tool
def add(a: int, b: int) -> int:
    """Сложение двух чисел."""
    return a + b


@tool
def delete_everything() -> str:
    """Опасный инструмент."""
    return "done"


def test_filter_tools_allows_only_allowlist():
    tools = [add, delete_everything]
    result = filter_tools(tools, {"add"})
    assert [t.name for t in result] == ["add"]


def test_filter_tools_empty_allowlist_returns_nothing():
    tools = [add, delete_everything]
    assert filter_tools(tools, set()) == []
