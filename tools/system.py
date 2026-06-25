"""Commandes système sécurisées."""

import subprocess

from loguru import logger

ALLOWED_COMMANDS = [
    "ls",
    "pwd",
    "echo",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "date",
    "uptime",
    "df",
    "free",
    "ps",
    "whoami",
    "hostname",
    "ping",
    "curl",
    "wget",
    "git status",
    "git log",
    "python3 --version",
    "node --version",
    "npm --version",
    "ollama list",
    "ollama ps",
]


def _is_command_allowed(commande: str) -> bool:
    cmd = commande.strip()
    for allowed in ALLOWED_COMMANDS:
        if cmd == allowed or cmd.startswith(allowed + " "):
            return True
    return False


def commande_systeme(commande: str) -> str:
    """Exécute une commande terminal sécurisée (whitelist)."""
    try:
        if not commande.strip():
            return "Veuillez spécifier une commande."

        if not _is_command_allowed(commande):
            return (
                f"Commande non autorisée : '{commande}'. "
                f"Commandes permises : {', '.join(ALLOWED_COMMANDS[:10])}..."
            )

        result = subprocess.run(
            commande,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (result.stdout + result.stderr).strip()
        if not output:
            output = f"Commande exécutée (code {result.returncode}), pas de sortie."
        if len(output) > 2000:
            output = output[:2000] + "\n... (tronqué à 2000 caractères)"
        return output
    except subprocess.TimeoutExpired:
        return "La commande a dépassé le délai de 15 secondes."
    except Exception as exc:
        logger.exception("Erreur commande_systeme")
        return f"Erreur exécution commande : {exc}"
