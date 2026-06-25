"""Tests STT."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_whisper_model():
    segment = MagicMock()
    segment.text = " Bonjour JARVIS"
    model = MagicMock()
    model.transcribe.return_value = ([segment], None)
    return model


@patch("faster_whisper.WhisperModel")
def test_transcribe_returns_text(mock_whisper_class, mock_whisper_model):
    from core.stt import STTEngine

    mock_whisper_class.return_value = mock_whisper_model
    engine = STTEngine()
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio)
    assert "Bonjour" in result


@patch("faster_whisper.WhisperModel")
def test_transcribe_empty_audio(mock_whisper_class, mock_whisper_model):
    from core.stt import STTEngine

    mock_whisper_class.return_value = mock_whisper_model
    engine = STTEngine()
    result = engine.transcribe(np.array([], dtype=np.float32))
    assert result == ""
