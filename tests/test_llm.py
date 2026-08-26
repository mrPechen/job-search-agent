import pytest

from config import settings
from src.llm.providers import build_text_model, build_vision_model


def test_build_text_model_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_text_model("unknown", "m", "k", "")


def test_build_vision_model_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_vision_model("unknown", "m", "k", "")


def test_gateway_exposes_embedder():
    """gateway.embedder собирается через build_embeddings и не падает."""
    from src.llm.gateway import LLMGateway

    gateway = LLMGateway()
    embedder = gateway.embedder
    assert embedder is not None
    assert embedder.model == settings.EMBEDDING_MODEL
