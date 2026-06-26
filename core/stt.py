"""Speech-To-Text avec faster-whisper."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import settings

_stt_instance: Optional["STTEngine"] = None


class STTEngine:
    def __init__(self) -> None:
        self._model = None
        self._load_model()

    def _load_model(self, retries: int = 3) -> None:
        from faster_whisper import WhisperModel

        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    f"Chargement Whisper {settings.whisper_model_size} "
                    f"({settings.whisper_device})..."
                )
                self._model = WhisperModel(
                    settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
                logger.info("Modèle Whisper chargé")
                return
            except Exception as exc:
                last_error = exc
                logger.warning(f"Tentative {attempt}/{retries} échouée : {exc}")
                time.sleep(1)

        raise RuntimeError(f"Impossible de charger Whisper : {last_error}")

    def transcribe(self, audio_array: np.ndarray) -> str:
        """Transcrit un numpy array float32 16kHz mono."""
        if self._model is None:
            self._load_model()

        if audio_array is None or len(audio_array) == 0:
            return ""

        start = time.perf_counter()
        try:
            segments, _ = self._model.transcribe(
                audio_array,
                language=settings.whisper_language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=(
                    f"Bonjour. Hey {settings.jarvis_name}, qu'est-ce qu'une API, "
                    "développement logiciel, intelligence artificielle."
                ),
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            duration = time.perf_counter() - start
            if text and len(text.strip(".… ")) >= 2:
                logger.info(f"Transcription ({duration:.2f}s) : {text[:80]}...")
            else:
                logger.debug(f"Transcription vide ({duration:.2f}s)")
            return text
        except Exception:
            logger.exception("Erreur transcription, nouvelle tentative...")
            try:
                self._load_model()
                segments, _ = self._model.transcribe(
                    audio_array,
                    language=settings.whisper_language,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                duration = time.perf_counter() - start
                logger.info(f"Transcription retry ({duration:.2f}s) : {text[:80]}...")
                return text
            except Exception as retry_exc:
                logger.exception("Échec définitif transcription")
                raise RuntimeError(f"Transcription impossible : {retry_exc}") from retry_exc

    def transcribe_file(self, path: str) -> str:
        """Transcrit un fichier audio."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier audio introuvable : {path}")

        start = time.perf_counter()
        try:
            segments, _ = self._model.transcribe(
                str(file_path),
                language=settings.whisper_language,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            duration = time.perf_counter() - start
            logger.info(f"Transcription fichier ({duration:.2f}s) : {file_path.name}")
            return text
        except Exception:
            logger.exception(f"Erreur transcription fichier {path}")
            raise


def get_stt_engine() -> STTEngine:
    """Singleton STT — charge le modèle une seule fois."""
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = STTEngine()
    return _stt_instance
