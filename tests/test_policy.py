from src.agent.policy import classify_action, requires_human_approval


def test_classify_action_maps_known_actions():
    assert classify_action("search") == "read"
    assert classify_action("draft_reply") == "draft"
    assert classify_action("apply") == "high_risk"
    assert classify_action("send_message") == "high_risk"


def test_classify_action_unknown_defaults_to_read():
    assert classify_action("whatever") == "read"


def test_requires_human_approval_only_high_risk():
    assert requires_human_approval("apply") is True
    assert requires_human_approval("send_message") is True
    assert requires_human_approval("draft_reply") is False
    assert requires_human_approval("search") is False
