"""Outils date, heure, timer et alarme."""

import threading
from datetime import datetime, timedelta
from typing import Any

import schedule
from dateutil import parser as date_parser
from loguru import logger

_timers: dict[str, dict[str, Any]] = {}
_timer_lock = threading.Lock()


def heure_date(action: str = "maintenant") -> str:
    """
    Donne l'heure, la date ou calcule des durées.
    action: 'maintenant', 'date', 'heure', ou une date à parser.
    """
    try:
        now = datetime.now()
        action_lower = action.lower().strip()

        if action_lower in ("maintenant", "now", ""):
            return (
                f"Nous sommes le {now.strftime('%A %d %B %Y')} "
                f"et il est {now.strftime('%H:%M')}."
            )
        if action_lower in ("date", "jour"):
            return f"Aujourd'hui nous sommes le {now.strftime('%A %d %B %Y')}."
        if action_lower in ("heure", "time"):
            return f"Il est {now.strftime('%H:%M')}."

        try:
            target = date_parser.parse(action, dayfirst=True)
            delta = target - now
            if delta.total_seconds() > 0:
                days = delta.days
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                minutes = remainder // 60
                return (
                    f"Il reste {days} jour(s), {hours} heure(s) et "
                    f"{minutes} minute(s) avant le {target.strftime('%d/%m/%Y %H:%M')}."
                )
            return f"La date {target.strftime('%d/%m/%Y %H:%M')} est passée."
        except (ValueError, TypeError):
            return (
                f"Nous sommes le {now.strftime('%A %d %B %Y')} "
                f"à {now.strftime('%H:%M')}."
            )
    except Exception as exc:
        logger.exception("Erreur heure_date")
        return f"Erreur lors du calcul de date/heure : {exc}"


def _run_timer_callback(timer_id: str, label: str) -> None:
    logger.info(f"Timer '{label}' (#{timer_id}) terminé !")
    with _timer_lock:
        if timer_id in _timers:
            _timers[timer_id]["done"] = True


def timer(duree: str, label: str = "minuteur") -> str:
    """
    Crée un minuteur. duree: ex '5 minutes', '30 secondes', '1 heure'.
    """
    try:
        duree_lower = duree.lower().strip()
        seconds = 0

        if "seconde" in duree_lower or "sec" in duree_lower:
            num = int("".join(c for c in duree_lower.split()[0] if c.isdigit()) or "0")
            seconds = num
        elif "minute" in duree_lower or "min" in duree_lower:
            num = int("".join(c for c in duree_lower.split()[0] if c.isdigit()) or "0")
            seconds = num * 60
        elif "heure" in duree_lower or "h" in duree_lower:
            num = int("".join(c for c in duree_lower.split()[0] if c.isdigit()) or "0")
            seconds = num * 3600
        else:
            try:
                seconds = int(duree_lower)
            except ValueError:
                return f"Durée non reconnue : '{duree}'. Utilisez ex: '5 minutes'."

        if seconds <= 0:
            return "La durée doit être positive."

        timer_id = f"timer_{len(_timers) + 1}"
        end_time = datetime.now() + timedelta(seconds=seconds)

        with _timer_lock:
            _timers[timer_id] = {
                "label": label,
                "seconds": seconds,
                "end_time": end_time.isoformat(),
                "done": False,
            }

        def job() -> None:
            _run_timer_callback(timer_id, label)

        schedule.every(seconds).seconds.do(job).tag(timer_id)

        def scheduler_loop() -> None:
            import time

            while True:
                schedule.run_pending()
                time.sleep(1)

        thread = threading.Thread(target=scheduler_loop, daemon=True)
        thread.start()

        return (
            f"Minuteur '{label}' de {seconds} secondes lancé. "
            f"Fin prévue à {end_time.strftime('%H:%M:%S')}."
        )
    except Exception as exc:
        logger.exception("Erreur timer")
        return f"Erreur création minuteur : {exc}"
