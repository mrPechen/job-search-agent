from src.agent.router import keyword_classify


def test_keyword_classify_search():
    assert keyword_classify("посмотри что нового по работе").intent == "search_job"


def test_keyword_classify_stats():
    assert keyword_classify("сколько откликов я сделал").intent == "stats"


def test_keyword_classify_confirm():
    assert keyword_classify("да, отправляй").intent == "confirm"


def test_keyword_classify_chat():
    assert keyword_classify("привет, как дела?").intent == "chat"
