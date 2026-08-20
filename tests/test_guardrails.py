from src.agent.guardrails import check_outgoing_message


def test_empty_message_blocked():
    ok, reason = check_outgoing_message("   ")
    assert ok is False
    assert reason


def test_secret_leak_blocked():
    ok, reason = check_outgoing_message(
        "мой api-ключ: sk-abcdefghijklmnopqrstuvwxyz123456"
    )
    assert ok is False
    assert "секрет" in reason


def test_normal_message_allowed():
    ok, reason = check_outgoing_message("Здравствуйте! Готов обсудить детали вакансии.")
    assert ok is True
    assert reason == ""


def test_too_long_message_blocked():
    ok, _ = check_outgoing_message("а" * 5000)
    assert ok is False
