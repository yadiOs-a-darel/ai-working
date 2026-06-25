"""Capture microphone et lecture audio."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn

from config.settings import settings

console = Console()

RMS_THRESHOLD = 0.01
SILENCE_FRAMES_REQUIRED = 3


class AudioCapture:
    def __init__(self) -> None:
        self.sample_rate = settings.audio_sample_rate
        self.channels = settings.audio_channels
        self.chunk_duration = settings.audio_chunk_duration
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = settings.vad_silence_threshold
        self.device = (
            None if settings.mic_device_index < 0 else settings.mic_device_index
        )
        self.last_speech_detected = False

    def _rms(self, chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))

    def record_until_silence(self, max_duration: float = 30.0) -> np.ndarray:
        """Enregistre depuis le micro jusqu'à détection de silence."""
        max_chunks = int(max_duration / self.chunk_duration)
        silence_frames_needed = max(
            1, int(self.silence_threshold / self.chunk_duration)
        )
        min_chunks_before_stop = max(
            SILENCE_FRAMES_REQUIRED,
            int(1.0 / self.chunk_duration),
        )
        recorded: list[np.ndarray] = []
        silence_count = 0
        speech_detected = False
        chunks_recorded = 0
        self.last_speech_detected = False

        progress = Progress(
            TextColumn("[cyan]🎤 Enregistrement"),
            BarColumn(bar_width=40),
            TextColumn("[green]{task.fields[level]:.2f}"),
            console=console,
            transient=True,
        )

        try:
            with Live(progress, refresh_per_second=10, console=console):
                task = progress.add_task("record", total=max_chunks, level=0.0)
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.chunk_samples,
                    device=self.device,
                ) as stream:
                    while chunks_recorded < max_chunks:
                        chunk, _ = stream.read(self.chunk_samples)
                        chunk = chunk.flatten().astype(np.float32)
                        recorded.append(chunk)
                        chunks_recorded += 1

                        level = self._rms(chunk)
                        progress.update(task, advance=1, level=level)

                        if level >= RMS_THRESHOLD:
                            speech_detected = True
                            silence_count = 0
                        elif speech_detected:
                            silence_count += 1
                            if (
                                silence_count >= silence_frames_needed
                                and len(recorded) >= min_chunks_before_stop
                            ):
                                break
        except Exception:
            logger.exception("Erreur lors de l'enregistrement audio")
            raise

        if not recorded:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(recorded)
        self.last_speech_detected = speech_detected
        logger.debug(
            f"Audio enregistré : {len(audio) / self.sample_rate:.1f}s "
            f"(parole détectée : {speech_detected})"
        )
        return audio

    def play_beep(self, frequency: int = 880, duration: float = 0.15) -> None:
        """Joue un bip court de confirmation."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        try:
            sd.play(tone, self.sample_rate)
            sd.wait()
            sd.sleep(0.25)
        except Exception:
            logger.warning("Impossible de jouer le bip de confirmation")


def play_audio(audio_path: str) -> None:
    """Joue un fichier audio (mp3/wav — lecteurs système en priorité)."""
    path = Path(audio_path)
    if not path.exists():
        logger.error(f"Fichier audio introuvable : {audio_path}")
        return

    suffix = path.suffix.lower()

    if suffix == ".mp3" and shutil.which("mpg123"):
        try:
            subprocess.run(["mpg123", "-q", str(path)], check=True, timeout=120)
            return
        except Exception:
            logger.warning("Lecture mpg123 échouée, essai suivant")

    if shutil.which("paplay"):
        try:
            subprocess.run(["paplay", str(path)], check=True, timeout=120)
            return
        except Exception:
            logger.warning("Lecture paplay échouée, essai suivant")

    if suffix in (".wav", ".flac", ".ogg"):
        try:
            data, samplerate = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            sd.play(data, samplerate)
            sd.wait()
            return
        except Exception:
            logger.warning("Lecture sounddevice échouée, fallback lecteur externe")

    player = None
    if suffix == ".mp3" and shutil.which("ffplay"):
        player = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
    elif shutil.which("aplay"):
        player = ["aplay", "-q", str(path)]
    elif shutil.which("ffplay"):
        player = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]

    if not player:
        logger.error("Aucun lecteur audio trouvé (paplay, mpg123, aplay, ffplay)")
        return

    try:
        subprocess.run(player, check=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.error("Timeout lecture audio")
    except subprocess.CalledProcessError:
        logger.exception(f"Erreur lecture audio : {audio_path}")
