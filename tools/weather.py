"""Outil météo."""

import requests
from loguru import logger

from config.settings import settings


def meteo(ville: str = "") -> str:
    """Obtient la météo actuelle pour une ville."""
    location = ville.strip() or settings.user_location.split(",")[0].strip()

    if settings.openweather_api_key:
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": settings.openweather_api_key,
                "units": "metric",
                "lang": "fr",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            temp = round(data["main"]["temp"])
            description = data["weather"][0]["description"]
            wind = round(data["wind"]["speed"] * 3.6)
            humidity = data["main"]["humidity"]
            return (
                f"À {location} : {temp}°C, {description}. "
                f"Vent {wind} km/h, humidité {humidity}%."
            )
        except Exception:
            logger.exception("Erreur OpenWeatherMap, fallback wttr.in")

    try:
        response = requests.get(
            f"https://wttr.in/{location}?format=3",
            timeout=10,
            headers={"User-Agent": "JARVIS/1.0"},
        )
        response.raise_for_status()
        wttr_text = response.text.strip()
        return f"Météo à {location} : {wttr_text}"
    except Exception as exc:
        logger.exception("Erreur météo wttr.in")
        return f"Impossible d'obtenir la météo pour {location} : {exc}"
