"""Détection du mot-clé d'activation."""

import re
import unicodedata
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from config.settings import settings

_wake_word_instance: Optional["WakeWordDetector"] = None

_WAKE_GREETING_PREFIXES = (
    "hey", "ok", "salut", "bonjour", "bonsoir", "coucou", "dis", "dit",
)


def _normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _is_meaningful_transcript(text: str) -> bool:
    cleaned = text.strip().strip(".…")
    if len(cleaned) < 2:
        return False
    if cleaned.replace(".", "").replace(" ", "") == "":
        return False
    return len(cleaned.split()) >= 1


class WakeWordDetector:
    def __init__(self) -> None:
        self.wake_word = settings.wake_word.lower()
        self.sensitivity = settings.wake_word_sensitivity
        self.sample_rate = settings.audio_sample_rate
        self._phrases = self._load_phrases()
        self._name_token = _normalize_phrase(settings.jarvis_name)
        self._pending_command: str = ""
        self._model = None
        self._active = False
        self._init_model()

    def _load_phrases(self) -> list[str]:
        raw = settings.wake_word_phrases or settings.wake_word
        phrases = [_normalize_phrase(p) for p in raw.split(",") if p.strip()]
        primary = _normalize_phrase(settings.wake_word)
        if primary and primary not in phrases:
            phrases.insert(0, primary)
        return phrases

    def pop_pending_command(self) -> str:
        """Commande prononcée dans la même phrase que le wake word."""
        cmd = self._pending_command.strip()
        self._pending_command = ""
        return cmd

    def _init_model(self) -> None:
        wake_normalized = _normalize_phrase(settings.wake_word)
        if "jarvis" not in wake_normalized:
            logger.info(
                f"Écoute vocale STT : « {settings.wake_word} » "
                f"(dites le nom {settings.jarvis_name})"
            )
            self._model = None
            return
        try:
            from openwakeword.model import Model

            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            logger.info(f"Wake word OpenWakeWord actif : \"{settings.wake_word}\"")
        except Exception:
            logger.warning(
                f"OpenWakeWord indisponible — écoute vocale « {settings.wake_word} »"
            )
            self._model = None

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def _contains_assistant_name(self, normalized: str) -> bool:
        if not normalized:
            return False
        if self._name_token and self._name_token in normalized:
            return True
        compact = normalized.replace(" ", "")
        return "yadjios" in compact or "yadji" in normalized.split()

    def _extract_command(self, normalized: str) -> str:
        """Texte après le nom de l'assistant dans la même phrase."""
        if not self._contains_assistant_name(normalized):
            return ""

        for phrase in sorted(self._phrases, key=len, reverse=True):
            if normalized.startswith(phrase):
                rest = normalized[len(phrase) :].strip(" ,.?!")
                if len(rest.split()) >= 2:
                    return rest

        if self._name_token and self._name_token in normalized:
            idx = normalized.find(self._name_token)
            rest = normalized[idx + len(self._name_token) :].strip(" ,.?!")
        else:
            match = re.search(r"yadji\s*os?", normalized)
            if not match:
                return ""
            rest = normalized[match.end() :].strip(" ,.?!")

        if len(rest.split()) >= 2:
            return rest
        return ""

    def parse_wake_transcript(self, transcript: str) -> tuple[bool, str]:
        """Retourne (wake détecté, commande éventuelle dans la même phrase)."""
        normalized = _normalize_phrase(transcript)
        if not normalized:
            return False, ""

        for phrase in self._phrases:
            if phrase in normalized or normalized.startswith(phrase):
                return True, self._extract_command(normalized)

        if self._contains_assistant_name(normalized):
            for prefix in _WAKE_GREETING_PREFIXES:
                if normalized.startswith(prefix) or f"{prefix} " in normalized:
                    return True, self._extract_command(normalized)
            # « YadjiOs, qu'est-ce qu'une API ? »
            if normalized.startswith(self._name_token or "yadji"):
                return True, self._extract_command(normalized)
            # Nom présent n'importe où : « dis à YadjiOs ... » ou phrase mixte
            return True, self._extract_command(normalized)

        return False, ""

    def detect_in_audio(self, audio_chunk: np.ndarray) -> bool:
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

    def _record_phrase(self, max_duration: float = 6.0) -> np.ndarray:
        """Capture une phrase (parole puis silence)."""
        chunk_duration = settings.audio_chunk_duration
        chunk_samples = int(self.sample_rate * chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        device = None if settings.mic_device_index < 0 else settings.mic_device_index

        recorded: list[np.ndarray] = []
        silence_count = 0
        speech_detected = False
        silence_needed = max(1, int(settings.vad_silence_threshold / chunk_duration))
        rms_threshold = 0.012
        min_speech_chunks = max(2, int(1.0 / chunk_duration))

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=device,
        ) as stream:
            speech_chunks = 0
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.flatten().astype(np.float32)
                recorded.append(chunk)
                level = float(np.sqrt(np.mean(np.square(chunk))))
                if level >= rms_threshold:
                    speech_detected = True
                    silence_count = 0
                    speech_chunks += 1
                elif speech_detected:
                    silence_count += 1
                    if (
                        silence_count >= silence_needed
                        and speech_chunks >= min_speech_chunks
                    ):
                        break

        if not recorded or not speech_detected or speech_chunks < min_speech_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(recorded)

    def _wait_stt_wake_word(self) -> bool:
        from core.stt import get_stt_engine

        stt = get_stt_engine()
        logger.info(
            f"Écoute active — dites « {settings.wake_word} » ou "
            f"« Bonjour {settings.jarvis_name} » (Ctrl+C pour quitter)"
        )

        while self._active:
            audio = self._record_phrase(max_duration=6.0)
            if len(audio) == 0:
                continue
            try:
                transcript = stt.transcribe(audio)
            except Exception:
                logger.exception("Erreur transcription wake word")
                continue
            if not _is_meaningful_transcript(transcript):
                logger.debug(f"Bruit ignoré : {transcript!r}")
                continue

            logger.info(f"Écoute : {transcript[:100]}")
            matched, command = self.parse_wake_transcript(transcript)
            if matched:
                self._pending_command = command
                logger.info(
                    f"Wake word détecté : {transcript[:80]}"
                    + (f" → commande : {command[:60]}" if command else "")
                )
                return True

        return False

    def wait_for_wake_word(
        self,
        callback: Optional[Callable[[], None]] = None,
        debug_stdin: bool = False,
    ) -> bool:
        if debug_stdin:
            try:
                input("\n[DEBUG] Appuyez sur Entrée pour parler... ")
                return True
            except EOFError:
                return False

        self._active = True
        self._pending_command = ""

        if self._model is not None:
            chunk_samples = int(self.sample_rate * settings.audio_chunk_duration)
            device = None if settings.mic_device_index < 0 else settings.mic_device_index
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
                logger.exception("Erreur écoute wake word OpenWakeWord")
            return False

        detected = self._wait_stt_wake_word()
        if detected and callback:
            callback()
        return detected

    def stop(self) -> None:
        self._active = False


def get_wake_word_detector() -> WakeWordDetector:
    global _wake_word_instance
    if _wake_word_instance is None:
        _wake_word_instance = WakeWordDetector()
    return _wake_word_instance
