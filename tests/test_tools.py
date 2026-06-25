"""Tests outils."""

from unittest.mock import MagicMock, patch

from tools.datetime_tool import heure_date, timer
from tools.files import creer_fichier, lister_fichiers, lire_fichier
from tools.system import commande_systeme
from tools.registry import get_all_tools


def test_heure_date_maintenant():
    result = heure_date("maintenant")
    assert "sommes" in result.lower() or "est" in result.lower()


def test_timer_creation():
    result = timer("5 secondes", "test")
    assert "minuteur" in result.lower() or "secondes" in result.lower()


def test_lister_fichiers_forbidden():
    result = lister_fichiers("/etc")
    assert "refusé" in result.lower() or "protégé" in result.lower()


def test_commande_systeme_allowed():
    result = commande_systeme("date")
    assert len(result) > 0


def test_commande_systeme_denied():
    result = commande_systeme("rm -rf /")
    assert "non autorisée" in result.lower()


def test_get_all_tools_count():
    tools = get_all_tools()
    assert len(tools) >= 11
    names = [t.name for t in tools]
    assert "recherche_web" in names
    assert "meteo" in names


@patch("tools.weather.requests.get")
def test_meteo_wttr_fallback(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        text="Bangui: ☀️ +32°C",
    )
    from tools.weather import meteo

    result = meteo("Bangui")
    assert "Bangui" in result or "32" in result
