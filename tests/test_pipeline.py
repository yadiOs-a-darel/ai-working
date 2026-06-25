"""Tests pipeline."""

from unittest.mock import MagicMock, patch


@patch("core.pipeline.get_jarvis_agent")
@patch("core.pipeline.get_stt_engine")
@patch("core.pipeline.get_tts_engine")
@patch("core.pipeline.get_wake_word_detector")
@patch("core.pipeline.get_long_term_memory")
def test_process_text(
    mock_ltm,
    mock_wake,
    mock_tts,
    mock_stt,
    mock_agent,
):
    from core.pipeline import JarvisPipeline

    agent_instance = MagicMock()
    agent_instance.chat.return_value = "Bonjour Darel"
    mock_agent.return_value = agent_instance

    pipeline = JarvisPipeline(debug=True)
    result = pipeline.process_text("Salut JARVIS comment vas-tu")
    assert result == "Bonjour Darel"
    agent_instance.chat.assert_called_once()


@patch("core.pipeline.get_jarvis_agent")
@patch("core.pipeline.get_stt_engine")
@patch("core.pipeline.get_tts_engine")
@patch("core.pipeline.get_wake_word_detector")
@patch("core.pipeline.get_long_term_memory")
def test_process_text_empty(
    mock_ltm,
    mock_wake,
    mock_tts,
    mock_stt,
    mock_agent,
):
    from core.pipeline import JarvisPipeline

    pipeline = JarvisPipeline(debug=True)
    result = pipeline.process_text("")
    assert result == ""
