"""Boucle principale voix-à-voix JARVIS."""

import asyncio
from typing import Optional

from loguru import logger
from rich.console import Console

from agent.jarvis_agent import get_jarvis_agent
from config.settings import settings
from core.audio import AudioCapture
from core.stt import get_stt_engine
from core.tts import get_tts_engine
from core.wake_word import get_wake_word_detector
from memory.long_term import get_long_term_memory

console = Console()

_pipeline_instance: Optional["JarvisPipeline"] = None


class JarvisPipeline:
    def __init__(self, debug: bool = False, text_only: bool = False) -> None:
        self.debug = debug
        self.text_only = text_only
        self.audio = None if text_only else AudioCapture()
        self.stt = None if text_only else get_stt_engine()
        self.agent = get_jarvis_agent(minimal=text_only)
        self.tts = None if text_only else get_tts_engine()
        self.wake_word = None if text_only else get_wake_word_detector()
        self.long_term = None if text_only else get_long_term_memory()
        self.is_running = False

    def process_text(
        self, text: str, vocal_mode: bool = False, speak: bool | None = None,
        stream: bool = False,
    ) -> str:
        """Traite une entrée texte et lit la réponse si demandé."""
        if not text.strip():
            return ""
        if stream and self.text_only:
            parts: list[str] = []
            for token in self.agent.chat_stream_tokens(text, vocal_mode=vocal_mode):
                console.print(token, end="", highlight=False)
                parts.append(token)
            console.print()
            response = "".join(parts)
        else:
            response = self.agent.chat(text, vocal_mode=vocal_mode)
        should_speak = speak if speak is not None else settings.speak_responses
        if should_speak and response.strip():
            self.tts.speak(response)
        return response

    def _handle_interaction(self, vocal: bool = True) -> None:
        """Gère un cycle complet d'interaction."""
        if vocal:
            self.audio.play_beep()
            audio = self.audio.record_until_silence()
            if len(audio) == 0:
                return
            text = self.stt.transcribe(audio)
        else:
            text = input("\n[Vous]: ").strip()

        if not text or len(text.split()) < 3:
            logger.debug("Texte trop court, retour en écoute")
            return

        console.print(f"[bold cyan][Vous]:[/bold cyan] {text}")
        response = self.agent.chat(text, vocal_mode=vocal)
        console.print(f"[bold green][{settings.jarvis_name}]:[/bold green] {response}")

        if vocal:
            self.tts.speak(response)

    def run(self) -> None:
        """Boucle principale : écoute → transcrit → raisonne → parle."""
        self.is_running = True
        logger.info("Pipeline vocal démarré")

        while self.is_running:
            try:
                detected = self.wake_word.wait_for_wake_word(debug_stdin=self.debug)
                if not detected:
                    break
                self._handle_interaction(vocal=True)
            except KeyboardInterrupt:
                logger.info("Arrêt demandé par l'utilisateur")
                break
            except Exception:
                logger.exception("Erreur dans la boucle pipeline")
                continue

        self.is_running = False
        logger.info("Pipeline vocal arrêté")

    async def run_async(self) -> None:
        """Version async pour usage avec FastAPI."""
        self.is_running = True
        loop = asyncio.get_event_loop()

        while self.is_running:
            try:
                detected = await loop.run_in_executor(
                    None,
                    self.wake_word.wait_for_wake_word,
                    None,
                    self.debug,
                )
                if not detected:
                    break
                await loop.run_in_executor(None, self._handle_interaction, True)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Erreur pipeline async")
                await asyncio.sleep(1)

        self.is_running = False

    def stop(self) -> None:
        self.is_running = False
        self.wake_word.stop()


def get_pipeline(debug: bool = False, text_only: bool = False) -> JarvisPipeline:
    global _pipeline_instance
    if _pipeline_instance is None or _pipeline_instance.text_only != text_only:
        _pipeline_instance = JarvisPipeline(debug=debug, text_only=text_only)
    return _pipeline_instance
