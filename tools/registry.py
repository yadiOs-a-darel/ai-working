"""Registre central des outils LangChain."""

from langchain_core.tools import StructuredTool
from loguru import logger

from memory.notes import get_notes_manager
from tools.calendar_tool import calendrier
from tools.datetime_tool import heure_date, timer
from tools.files import creer_fichier, lire_fichier, lister_fichiers
from tools.music import jouer_musique
from tools.system import commande_systeme
from tools.weather import meteo
from tools.web_search import recherche_web


def _prendre_note(contenu: str) -> str:
    try:
        note_id = get_notes_manager().add_note(contenu)
        return f"Note #{note_id} enregistrée avec succès."
    except Exception as exc:
        logger.exception("Erreur prendre_note")
        return f"Erreur enregistrement note : {exc}"


def _lire_notes() -> str:
    try:
        return get_notes_manager().get_notes_text()
    except Exception as exc:
        logger.exception("Erreur lire_notes")
        return f"Erreur lecture notes : {exc}"


def get_all_tools() -> list:
    """Retourne la liste complète des tools LangChain."""
    tools = [
        StructuredTool.from_function(
            func=recherche_web,
            name="recherche_web",
            description=(
                "Chercher des informations récentes sur internet. "
                "Utilise quand l'utilisateur demande des actualités ou des faits récents."
            ),
        ),
        StructuredTool.from_function(
            func=meteo,
            name="meteo",
            description=(
                "Obtenir la météo actuelle ou les prévisions pour une ville. "
                "Si aucune ville n'est précisée, utilise la localisation par défaut."
            ),
        ),
        StructuredTool.from_function(
            func=heure_date,
            name="heure_date",
            description=(
                "Donner l'heure, la date actuelle ou calculer une durée entre dates."
            ),
        ),
        StructuredTool.from_function(
            func=timer,
            name="timer",
            description=(
                "Créer un minuteur ou une alarme. "
                "Exemple de durée : '5 minutes', '30 secondes'."
            ),
        ),
        StructuredTool.from_function(
            func=_prendre_note,
            name="prendre_note",
            description="Sauvegarder une note rapide pour l'utilisateur.",
        ),
        StructuredTool.from_function(
            func=_lire_notes,
            name="lire_notes",
            description="Relire les notes enregistrées par l'utilisateur.",
        ),
        StructuredTool.from_function(
            func=lister_fichiers,
            name="lister_fichiers",
            description="Lister les fichiers et dossiers d'un répertoire donné.",
        ),
        StructuredTool.from_function(
            func=lire_fichier,
            name="lire_fichier",
            description="Lire le contenu d'un fichier texte.",
        ),
        StructuredTool.from_function(
            func=creer_fichier,
            name="creer_fichier",
            description="Créer ou modifier un fichier texte avec un contenu donné.",
        ),
        StructuredTool.from_function(
            func=commande_systeme,
            name="commande_systeme",
            description=(
                "Exécuter une commande terminal sécurisée "
                "(ls, pwd, date, git status, etc.)."
            ),
        ),
        StructuredTool.from_function(
            func=jouer_musique,
            name="jouer_musique",
            description=(
                "Démarrer, arrêter ou vérifier la lecture musicale. "
                "Actions : play, stop, status. Fournir un chemin ou URL pour play."
            ),
        ),
        StructuredTool.from_function(
            func=calendrier,
            name="calendrier",
            description="Consulter les événements du calendrier à venir.",
        ),
    ]
    logger.debug(f"{len(tools)} outils enregistrés")
    return tools
