"""Contrôle de la musique via VLC ou mpv."""

import shutil
import subprocess
from pathlib import Path

from loguru import logger

_player_process: subprocess.Popen | None = None


def _find_player() -> str | None:
    for player in ("mpv", "cvlc", "vlc"):
        if shutil.which(player):
            return player
    return None


def jouer_musique(action: str, source: str = "") -> str:
    """
    Contrôle la lecture musicale.
    action: 'play', 'stop', 'pause', 'status'
    source: chemin fichier ou URL pour play
    """
    global _player_process

    try:
        action_lower = action.lower().strip()

        if action_lower == "stop":
            if _player_process and _player_process.poll() is None:
                _player_process.terminate()
                _player_process = None
                return "Lecture musicale arrêtée."
            return "Aucune lecture en cours."

        if action_lower == "status":
            if _player_process and _player_process.poll() is None:
                return "Lecture musicale en cours."
            return "Aucune lecture en cours."

        if action_lower in ("play", "jouer", "lire"):
            if not source.strip():
                return "Veuillez fournir un fichier ou une URL à jouer."

            player = _find_player()
            if not player:
                return "Aucun lecteur trouvé (mpv ou VLC requis)."

            if _player_process and _player_process.poll() is None:
                _player_process.terminate()

            source_path = Path(source).expanduser()
            target = str(source_path) if source_path.exists() else source

            if player == "mpv":
                cmd = ["mpv", "--no-video", "--really-quiet", target]
            else:
                cmd = ["cvlc", "--play-and-exit", target] if player == "cvlc" else ["vlc", "--play-and-exit", target]

            _player_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Lecture démarrée : {target}"

        if action_lower in ("pause", "pause"):
            return "Pause non supportée en mode CLI. Utilisez 'stop'."

        return f"Action non reconnue : {action}. Utilisez play, stop ou status."
    except Exception as exc:
        logger.exception("Erreur jouer_musique")
        return f"Erreur contrôle musique : {exc}"
