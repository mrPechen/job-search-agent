import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from src.browser.server import build_browser_app

MOCK_DIR = Path(__file__).parent / "mock_site"


@pytest.fixture
def mock_site():
    """Поднять локальный HTTP-сервер, раздающий mock job-сайт, на свободном порту."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(MOCK_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}"
    server.shutdown()
    thread.join()
    server.server_close()


@pytest_asyncio.fixture
async def browser_client(tmp_path):
    """Собрать browser-приложение in-process и вернуть HTTP-клиент к нему."""
    app = build_browser_app(profiles_dir=tmp_path / "profiles", headless=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    # Освобождаем браузерные профили после каждого теста
    await app.state.browser_manager.close()


async def test_navigate_rejects_non_whitelisted_domain(browser_client):
    r = await browser_client.post(
        "/navigate", json={"user_id": 1, "url": "https://evil.com"}
    )
    assert r.status_code == 403


async def test_extract_finds_interactive_elements(mock_site, browser_client):
    await browser_client.post(
        "/navigate", json={"user_id": 1, "url": f"{mock_site}/index.html"}
    )
    r = await browser_client.post("/extract", json={"user_id": 1})
    data = r.json()

    texts = [el["text"] for el in data["elements"]]
    tags = [el["tag"] for el in data["elements"]]

    assert any("Откликнуться" in text for text in texts)
    assert "textarea" in tags


async def test_message_with_secret_is_blocked(browser_client):
    """Сообщение с утечкой секрета отклоняется гардрейлом (400)."""
    r = await browser_client.post(
        "/message",
        json={"user_id": 1, "text": "мой ключ sk-abcdefghijklmnopqrstuvwxyz123456"},
    )
    assert r.status_code == 400


async def test_click_and_type_and_send(mock_site, browser_client):
    await browser_client.post(
        "/navigate", json={"user_id": 1, "url": f"{mock_site}/index.html"}
    )
    await browser_client.post("/click", json={"user_id": 1, "selector": "#apply-btn"})
    await browser_client.post(
        "/type",
        json={"user_id": 1, "selector": "#chat-input", "text": "Здравствуйте"},
    )
    await browser_client.post("/click", json={"user_id": 1, "selector": "#send-btn"})

    r = await browser_client.post("/extract", json={"user_id": 1})
    assert "Здравствуйте" in r.json()["text"]


async def test_submit_application(mock_site, browser_client):
    """Отклик с сопроводительным письмом: /submit заполняет и отправляет форму."""
    await browser_client.post(
        "/navigate", json={"user_id": 1, "url": f"{mock_site}/index.html"}
    )
    r = await browser_client.post(
        "/submit", json={"user_id": 1, "cover_letter": "Готов приступить"}
    )
    assert r.status_code == 200

    r = await browser_client.post("/extract", json={"user_id": 1})
    assert "Готов приступить" in r.json()["text"]
