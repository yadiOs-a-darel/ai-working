"""Tests LLM."""

from unittest.mock import MagicMock, patch


@patch("core.llm.requests.get")
def test_ollama_available(mock_get):
    mock_get.return_value = MagicMock(status_code=200)
    from core.llm import LLMClient

    client = LLMClient()
    assert client.is_ollama_available() is True


@patch("core.llm.requests.get")
def test_ollama_unavailable(mock_get):
    mock_get.side_effect = ConnectionError("refused")
    from core.llm import LLMClient

    client = LLMClient()
    assert client.is_ollama_available() is False


@patch("core.llm.requests.get")
def test_get_model_info(mock_get):
    mock_get.return_value = MagicMock(status_code=200)
    from core.llm import LLMClient

    client = LLMClient()
    info = client.get_model_info()
    assert "model" in info
    assert "provider" in info
    assert info["ollama_available"] is True
