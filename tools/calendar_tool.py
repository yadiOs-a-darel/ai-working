"""Outil calendrier local et Google optionnel."""

from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from config.settings import settings


def _parse_ical_events(ics_path: Path, days_ahead: int = 7) -> list[str]:
    try:
        from icalendar import Calendar

        content = ics_path.read_bytes()
        cal = Calendar.from_ical(content)
        now = datetime.now()
        end = now + timedelta(days=days_ahead)
        events = []

        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("dtstart")
            if dtstart is None:
                continue
            start = dtstart.dt
            if isinstance(start, datetime) and now <= start <= end:
                summary = str(component.get("summary", "Sans titre"))
                events.append(f"- {start.strftime('%d/%m %H:%M')} : {summary}")
        return events
    except ImportError:
        return []
    except Exception:
        logger.exception("Erreur parsing iCal")
        return []


def calendrier(action: str = "aujourd'hui") -> str:
    """
    Consulte le calendrier local ou Google.
    action: 'aujourd'hui', 'semaine', 'prochains'
    """
    try:
        local_ics = Path.home() / "calendar.ics"
        events: list[str] = []

        if local_ics.exists():
            events = _parse_ical_events(local_ics)

        creds_path = Path(settings.google_calendar_credentials_path)
        if creds_path.exists():
            try:
                events.extend(_get_google_events())
            except Exception:
                logger.exception("Erreur Google Calendar")

        if not events:
            return "Aucun événement trouvé dans les 7 prochains jours."

        return "Événements à venir :\n" + "\n".join(events[:10])
    except Exception as exc:
        logger.exception("Erreur calendrier")
        return f"Erreur consultation calendrier : {exc}"


def _get_google_events() -> list[str]:
    """Récupère les événements Google Calendar (optionnel)."""
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_service_account_file(
            settings.google_calendar_credentials_path,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        service = build("calendar", "v3", credentials=creds)
        now = datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = events_result.get("items", [])
        formatted = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "Sans titre")
            formatted.append(f"- {start} : {summary}")
        return formatted
    except ImportError:
        return []
    except Exception:
        logger.exception("Google Calendar non disponible")
        return []
