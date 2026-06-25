"""Route statut API."""

from fastapi import APIRouter

from core.llm import get_llm_client
from core.wake_word import get_wake_word_detector
from tools.registry import get_all_tools

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
async def status_endpoint() -> dict:
    llm = get_llm_client()
    model_info = llm.get_model_info()
    wake_word = get_wake_word_detector()

    return {
        "status": "ok",
        "ollama_available": model_info["ollama_available"],
        "model": model_info["model"],
        "provider": model_info["provider"],
        "wake_word_active": wake_word.is_active,
        "tools_count": len(get_all_tools()),
        "version": "1.0.0",
    }
