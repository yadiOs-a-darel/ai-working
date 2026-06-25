"""Text-To-Speech multi-moteur."""

import asyncio
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from core.audio import play_audio

_tts_instance: Optional["TTSEngine"] = None
# Moteur verrouillé après la 1re phrase réussie — évite edge puis espeak (voix différentes)
_session_engine: Optional[str] = None

FEMININE_EDGE_VOICES = frozenset({
    "fr-FR-DeniseNeural",
    "fr-FR-EloiseNeural",
    "fr-FR-VivienneMultilingualNeural",
    "fr-BE-CharlineNeural",
    "fr-CA-SylvieNeural",
    "fr-CH-ArianeNeural",
})
EDGE_RETRY_ATTEMPTS = 3
EDGE_RETRY_DELAY_S = 0.6

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
HTML_PATTERN = re.compile(r"<[^>]+>")
MARKDOWN_PATTERN = re.compile(r"[*_`#\[\]()>|~-]+")


def clean_text_for_tts(text: str) -> str:
    """Nettoie le texte avant synthèse vocale."""
    text = EMOJI_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)
    text = MARKDOWN_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text_for_tts(text: str, max_chars: int = 500) -> list[str]:
    """Découpe le texte aux ponctuations si trop long."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    sentences = re.split(r"(?<=[.!?;:,])\s+", text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i : i + max_chars])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


class TTSEngine:
    def __init__(self) -> None:
        self.engine = settings.tts_engine
        self.voice = self._resolve_edge_voice()
        self.rate = settings.tts_rate
        self._feminine_edge_only = (
            settings.tts_voice_gender == "female" and self.engine == "edge"
        )

    def _resolve_edge_voice(self) -> str:
        voice = settings.tts_voice
        if settings.tts_voice_gender != "female":
            return voice
        if voice in FEMININE_EDGE_VOICES:
            return voice
        if any(
            name in voice
            for name in ("Denise", "Eloise", "Vivienne", "Charline", "Sylvie", "Ariane")
        ):
            return voice
        logger.warning(
            f"Voix edge « {voice} » non féminine — bascule sur fr-FR-DeniseNeural"
        )
        return "fr-FR-DeniseNeural"

    def _lock_engine(self, engine: str) -> None:
        global _session_engine
        if _session_engine is None:
            _session_engine = engine
            label = self.voice if engine == "edge" else engine
            logger.info(f"Voix verrouillée pour la session : {engine} / {label}")

    def _engines_to_try(self) -> list[str]:
        global _session_engine
        if _session_engine:
            return [_session_engine]
        if self._feminine_edge_only:
            return ["edge"]
        if self.engine == "edge":
            return ["edge"]
        if self.engine == "local":
            return ["edge"]
        return [self.engine, "edge"]

    async def _synthesize_edge(self, text: str, output_path: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def _synthesize_edge_sync(self, text: str, output_path: str) -> None:
        last_error: Exception | None = None
        for attempt in range(1, EDGE_RETRY_ATTEMPTS + 1):
            try:
                asyncio.run(self._synthesize_edge(text, output_path))
                path = Path(output_path)
                if path.exists() and path.stat().st_size > 0:
                    return
                raise RuntimeError("Fichier edge-tts vide")
            except Exception as exc:
                last_error = exc
                if attempt < EDGE_RETRY_ATTEMPTS:
                    logger.warning(
                        f"edge-tts tentative {attempt}/{EDGE_RETRY_ATTEMPTS} : {exc}"
                    )
                    time.sleep(EDGE_RETRY_DELAY_S)
        raise RuntimeError(
            f"edge-tts indisponible après {EDGE_RETRY_ATTEMPTS} tentatives : {last_error}"
        ) from last_error

    def _play_chunk(self, text: str, engine: str) -> None:
        suffix = ".mp3" if engine == "edge" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        audio_path = tmp_path
        try:
            if engine == "edge":
                audio_path = str(Path(tmp_path).with_suffix(".mp3"))
                self._synthesize_edge_sync(text, audio_path)
            else:
                raise RuntimeError(f"Moteur non supporté : {engine}")
            play_audio(audio_path)
            self._lock_engine(engine)
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def synthesize(self, text: str, output_path: str) -> str:
        """Synthétise le texte vers un fichier audio."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            logger.warning("Texte vide après nettoyage TTS")
            return output_path
        target = str(Path(output_path).with_suffix(".mp3"))
        self._synthesize_edge_sync(cleaned, target)
        self._lock_engine("edge")
        return target

    def speak(self, text: str) -> None:
        """Synthétise et joue — une seule voix (Denise) pour toute la session."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return

        engines = self._engines_to_try()
        for chunk in split_text_for_tts(cleaned, max_chars=300):
            spoken = False
            for engine in engines:
                try:
                    self._play_chunk(chunk, engine)
                    spoken = True
                    break
                except Exception as exc:
                    logger.warning(f"TTS {engine} échoué : {exc}")
            if not spoken:
                logger.error(
                    "Impossible de lire à voix haute — vérifiez la connexion "
                    "(edge-tts / Denise) et mpg123 : sudo apt install mpg123"
                )


def get_tts_engine(*, reload: bool = False) -> TTSEngine:
    """Singleton TTS — le verrou de voix (_session_engine) survit au reload."""
    global _tts_instance
    if reload or _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance
