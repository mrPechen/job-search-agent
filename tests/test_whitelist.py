from src.browser.server import _is_allowed_url


def test_static_whitelist_rejects_unknown_domain():
    assert _is_allowed_url("https://evil.com") is False


def test_allowed_domains_overrides_static():
    assert _is_allowed_url("https://evil.com", ["evil.com"]) is True
    assert _is_allowed_url("https://hh.ru", ["evil.com"]) is False


def test_rejects_bare_tld_namespace():
    assert _is_allowed_url("https://anything.attacker.com", ["com"]) is False


def test_normalizes_case_and_trailing_dot():
    assert _is_allowed_url("https://example.com", ["EXAMPLE.COM."]) is True


def test_empty_allowed_domains_rejects_all():
    assert _is_allowed_url("https://example.com", []) is False


def test_rejects_non_http_scheme():
    assert _is_allowed_url("file:///etc/passwd", ["whatever.com"]) is False


def test_subdomain_allowed():
    assert _is_allowed_url("https://spb.hh.ru", ["hh.ru"]) is True
