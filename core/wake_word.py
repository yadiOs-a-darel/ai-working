"""Détection du mot-clé d'activation."""

import threading
from typing import Callable, Optional

import numpy as np
from loguru import logger

from config.settings import settings

_wake_word_instance: Optional["WakeWordDetector"] = None


class WakeWordDetector:
    def __init__(self) -> None:
        self.wake_word = settings.wake_word.lower()
        self.sensitivity = settings.wake_word_sensitivity
        self.sample_rate = settings.audio_sample_rate
        self._model = None
        self._active = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from openwakeword.model import Model

            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            logger.info(f"Wake word actif : \"{settings.wake_word}\"")
        except Exception:
            logger.warning(
                "OpenWakeWord non disponible, mode écoute texte/stdin uniquement"
            )
            self._model = None

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def detect_in_audio(self, audio_chunk: np.ndarray) -> bool:
        """Détecte le wake word dans un chunk audio."""
        if self._model is None or len(audio_chunk) == 0:
            return False
        try:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            prediction = self._model.predict(audio_int16)
            for scores in prediction.values():
                if scores > self.sensitivity:
                    return True
            return False
        except Exception:
            logger.exception("Erreur détection wake word")
            return False

    def wait_for_wake_word(
        self,
        callback: Optional[Callable[[], None]] = None,
        debug_stdin: bool = False,
    ) -> bool:
        """Bloque jusqu'à détection du wake word ou entrée stdin en debug."""
        if debug_stdin:
            try:
                input("\n[DEBUG] Appuyez sur Entrée pour parler... ")
                return True
            except EOFError:
                return False

        if self._model is None:
            logger.info("Wake word désactivé — appuyez sur Entrée pour parler")
            try:
                input()
                return True
            except EOFError:
                return False

        import sounddevice as sd

        chunk_samples = int(self.sample_rate * settings.audio_chunk_duration)
        device = None if settings.mic_device_index < 0 else settings.mic_device_index

        self._active = True
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=chunk_samples,
                device=device,
            ) as stream:
                while self._active:
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk.flatten()
                    if self.detect_in_audio(chunk):
                        logger.info("Wake word détecté !")
                        if callback:
                            callback()
                        return True
        except Exception:
            logger.exception("Erreur écoute wake word")
        return False

    def stop(self) -> None:
        self._active = False


def get_wake_word_detector() -> WakeWordDetector:
    global _wake_word_instance
    if _wake_word_instance is None:
        _wake_word_instance = WakeWordDetector()
    return _wake_word_instance
