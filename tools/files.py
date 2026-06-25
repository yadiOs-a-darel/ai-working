"""Opérations sur les fichiers."""

from pathlib import Path

from loguru import logger

FORBIDDEN_PREFIXES = ("/etc", "/sys", "/proc", "/boot", "/dev", "/root")


def _is_path_allowed(path: str) -> bool:
    resolved = str(Path(path).expanduser().resolve())
    for prefix in FORBIDDEN_PREFIXES:
        if resolved.startswith(prefix):
            return False
    return True


def lister_fichiers(dossier: str) -> str:
    """Liste les fichiers d'un dossier (max 50 entrées)."""
    try:
        if not _is_path_allowed(dossier):
            return "Accès refusé : ce dossier système est protégé."

        path = Path(dossier).expanduser()
        if not path.exists():
            return f"Dossier introuvable : {dossier}"
        if not path.is_dir():
            return f"Ce n'est pas un dossier : {dossier}"

        entries = sorted(path.iterdir())[:50]
        if not entries:
            return f"Le dossier {dossier} est vide."

        lines = [f"Contenu de {path} ({len(entries)} entrées) :"]
        for entry in entries:
            prefix = "📁" if entry.is_dir() else "📄"
            lines.append(f"  {prefix} {entry.name}")
        if len(list(path.iterdir())) > 50:
            lines.append("  ... (tronqué à 50 entrées)")
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("Erreur lister_fichiers")
        return f"Erreur lecture dossier : {exc}"


def lire_fichier(chemin: str) -> str:
    """Lit le contenu d'un fichier texte (max 10 000 caractères)."""
    try:
        if not _is_path_allowed(chemin):
            return "Accès refusé : ce fichier système est protégé."

        path = Path(chemin).expanduser()
        if not path.exists():
            return f"Fichier introuvable : {chemin}"
        if not path.is_file():
            return f"Ce n'est pas un fichier : {chemin}"

        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > 10000:
            return content[:10000] + "\n... (tronqué à 10 000 caractères)"
        return content
    except Exception as exc:
        logger.exception("Erreur lire_fichier")
        return f"Erreur lecture fichier : {exc}"


def creer_fichier(chemin: str, contenu: str) -> str:
    """Crée ou écrase un fichier texte."""
    try:
        if not _is_path_allowed(chemin):
            return "Accès refusé : impossible d'écrire dans ce chemin système."

        path = Path(chemin).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contenu, encoding="utf-8")
        return f"Fichier créé avec succès : {path}"
    except Exception as exc:
        logger.exception("Erreur creer_fichier")
        return f"Erreur création fichier : {exc}"
