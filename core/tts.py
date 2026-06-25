"""Text-To-Speech multi-moteur."""

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from core.audio import play_audio

_tts_instance: Optional["TTSEngine"] = None

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
        self.voice = settings.tts_voice
        self._pyttsx3_engine = None

    def _init_pyttsx3(self) -> None:
        if self._pyttsx3_engine is None:
            import pyttsx3

            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 170)

    async def _synthesize_edge(self, text: str, output_path: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def _synthesize_edge_sync(self, text: str, output_path: str) -> None:
        asyncio.run(self._synthesize_edge(text, output_path))

    def _synthesize_pyttsx3(self, text: str, output_path: str) -> None:
        self._init_pyttsx3()
        self._pyttsx3_engine.save_to_file(text, output_path)
        self._pyttsx3_engine.runAndWait()

    def _synthesize_coqui(self, text: str, output_path: str) -> None:
        try:
            from TTS.api import TTS

            tts = TTS(model_name="tts_models/fr/css10/vits", progress_bar=False)
            tts.tts_to_file(text=text, file_path=output_path)
        except ImportError:
            logger.warning("Coqui TTS non installé, fallback pyttsx3")
            self._synthesize_pyttsx3(text, output_path)

    def _synthesize_elevenlabs(self, text: str, output_path: str) -> None:
        import requests

        if not settings.elevenlabs_api_key or not settings.elevenlabs_voice_id:
            raise ValueError("Clés ElevenLabs non configurées")

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/"
            f"{settings.elevenlabs_voice_id}"
        )
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": settings.elevenlabs_api_key,
        }
        response = requests.post(
            url,
            json={"text": text, "model_id": "eleven_multilingual_v2"},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        Path(output_path).write_bytes(response.content)

    def synthesize(self, text: str, output_path: str) -> None:
        """Synthétise le texte vers un fichier audio."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            logger.warning("Texte vide après nettoyage TTS")
            return

        chunks = split_text_for_tts(cleaned)
        if len(chunks) == 1:
            self._synthesize_chunk(chunks[0], output_path)
            return

        # Plusieurs morceaux : synthétiser le premier chunk principal
        combined_path = Path(output_path)
        temp_files: list[Path] = []
        try:
            for i, chunk in enumerate(chunks):
                tmp = combined_path.parent / f"{combined_path.stem}_part{i}.wav"
                self._synthesize_chunk(chunk, str(tmp))
                temp_files.append(tmp)

            if len(temp_files) == 1:
                temp_files[0].rename(output_path)
            else:
                # Jouer séquentiellement via speak pour les chunks multiples
                import shutil

                shutil.copy(str(temp_files[0]), output_path)
        finally:
            for tmp in temp_files:
                if tmp.exists() and str(tmp) != output_path:
                    tmp.unlink(missing_ok=True)

    def _synthesize_chunk(self, text: str, output_path: str) -> None:
        try:
            if self.engine == "edge":
                self._synthesize_edge_sync(text, output_path)
            elif self.engine == "pyttsx3":
                self._synthesize_pyttsx3(text, output_path)
            elif self.engine == "coqui":
                self._synthesize_coqui(text, output_path)
            elif self.engine == "elevenlabs":
                self._synthesize_elevenlabs(text, output_path)
            else:
                self._synthesize_edge_sync(text, output_path)
        except Exception:
            logger.exception(f"Erreur TTS ({self.engine}), fallback edge-tts")
            self._synthesize_edge_sync(text, output_path)

    def speak(self, text: str) -> None:
        """Synthétise et joue immédiatement."""
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return

        chunks = split_text_for_tts(cleaned)
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self.synthesize(chunk, tmp_path)
                play_audio(tmp_path)
            except Exception:
                logger.exception("Erreur lecture réponse vocale")
            finally:
                Path(tmp_path).unlink(missing_ok=True)


def get_tts_engine() -> TTSEngine:
    """Singleton TTS."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance
