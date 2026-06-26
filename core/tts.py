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
_session_engine: Optional[str] = None
_edge_available: Optional[bool] = None

FEMININE_EDGE_VOICES = frozenset({
    "fr-FR-DeniseNeural",
    "fr-FR-EloiseNeural",
    "fr-FR-VivienneMultilingualNeural",
    "fr-BE-CharlineNeural",
    "fr-CA-SylvieNeural",
    "fr-CH-ArianeNeural",
})
EDGE_RETRY_ATTEMPTS = 2
EDGE_RETRY_DELAY_S = 0.5

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
    text = EMOJI_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)
    text = MARKDOWN_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text_for_tts(text: str, max_chars: int = 500) -> list[str]:
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
        self.local_voice = settings.tts_local_voice
        self.rate = settings.tts_rate

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
        return "fr-FR-DeniseNeural"

    def _lock_engine(self, engine: str) -> None:
        global _session_engine
        if _session_engine is None:
            _session_engine = engine
            if engine == "edge":
                label = self.voice
            elif engine == "espeak":
                label = self.local_voice
            else:
                label = engine
            logger.info(f"Voix verrouillée pour la session : {engine} / {label}")

    def _check_edge_available(self) -> bool:
        """Test rapide edge-tts (Microsoft bloque souvent avec 403)."""
        global _edge_available
        if _edge_available is not None:
            return _edge_available
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
        try:
            self._synthesize_edge_sync("ok", path, retries=1)
            _edge_available = True
            return True
        except Exception as exc:
            _edge_available = False
            logger.warning(
                f"edge-tts indisponible ({exc}) — repli voix féminine locale "
                f"espeak ({self.local_voice})"
            )
            return False
        finally:
            Path(path).unlink(missing_ok=True)

    def _engines_to_try(self) -> list[str]:
        global _session_engine
        if _session_engine:
            return [_session_engine]

        chain: list[str] = []
        if self.engine == "edge" and self._check_edge_available():
            chain.append("edge")
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            chain.append("espeak")
        if not chain:
            chain = ["edge"]
        return chain

    async def _synthesize_edge(self, text: str, output_path: str) -> None:
        import edge_tts

        await edge_tts.Communicate(text, self.voice).save(output_path)

    def _synthesize_edge_sync(
        self, text: str, output_path: str, retries: int = EDGE_RETRY_ATTEMPTS
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                asyncio.run(self._synthesize_edge(text, output_path))
                path = Path(output_path)
                if path.exists() and path.stat().st_size > 0:
                    return
                raise RuntimeError("Fichier edge-tts vide")
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(EDGE_RETRY_DELAY_S)
        raise RuntimeError(f"edge-tts : {last_error}") from last_error

    def _synthesize_espeak_ng(self, text: str, output_path: str) -> None:
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak:
            raise RuntimeError("espeak-ng introuvable — sudo apt install espeak-ng")
        subprocess.run(
            [espeak, "-v", self.local_voice, "-s", str(self.rate), "-w", output_path, text],
            check=True,
            timeout=120,
        )

    def _play_chunk(self, text: str, engine: str) -> None:
        suffix = ".mp3" if engine == "edge" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        audio_path = tmp_path
        try:
            if engine == "edge":
                audio_path = str(Path(tmp_path).with_suffix(".mp3"))
                self._synthesize_edge_sync(text, audio_path)
            elif engine == "espeak":
                self._synthesize_espeak_ng(text, audio_path)
            else:
                raise RuntimeError(f"Moteur inconnu : {engine}")
            play_audio(audio_path)
            self._lock_engine(engine)
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def speak(self, text: str) -> None:
        """Synthétise et joue — une voix verrouillée par session (edge ou espeak)."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return

        for chunk in split_text_for_tts(cleaned, max_chars=300):
            spoken = False
            for engine in self._engines_to_try():
                try:
                    self._play_chunk(chunk, engine)
                    spoken = True
                    break
                except Exception as exc:
                    logger.warning(f"TTS {engine} échoué : {exc}")
                    if engine == "edge":
                        global _edge_available
                        _edge_available = False
            if not spoken:
                logger.error(
                    "Impossible de lire à voix haute — installez : "
                    "sudo apt install espeak-ng mpg123"
                )


def get_tts_engine(*, reload: bool = False) -> TTSEngine:
    global _tts_instance, _edge_available
    if reload:
        _edge_available = None
    if reload or _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance
