"""Routes mémoire API."""

from fastapi import APIRouter

from memory.notes import get_notes_manager
from memory.short_term import ShortTermMemory

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/history")
async def get_history(session_id: str = "default") -> dict:
    memory = ShortTermMemory(session_id)
    return {"messages": memory.get_messages()}


@router.delete("/history")
async def clear_history(session_id: str = "default") -> dict:
    memory = ShortTermMemory(session_id)
    memory.clear()
    return {"status": "ok", "message": "Historique effacé"}


@router.get("/notes")
async def get_notes() -> dict:
    notes = get_notes_manager().get_all_notes()
    return {"notes": notes}
