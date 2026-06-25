"""Routes chat API."""

import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from agent.jarvis_agent import get_jarvis_agent

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str]
    duration_ms: int


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
    start = time.perf_counter()
    agent = get_jarvis_agent(request.session_id)
    response = agent.chat(request.message)
    duration_ms = int((time.perf_counter() - start) * 1000)
    return {
        "response": response,
        "tools_used": [],
        "duration_ms": duration_ms,
    }
