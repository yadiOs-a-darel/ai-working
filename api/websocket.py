"""WebSocket streaming temps réel."""

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from agent.jarvis_agent import get_jarvis_agent

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    agent = get_jarvis_agent()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload: dict[str, Any] = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "content": "JSON invalide"}
                )
                continue

            msg_type = payload.get("type", "message")
            if msg_type != "message":
                continue

            content = payload.get("content", "").strip()
            if not content:
                continue

            full_response = ""
            for token in agent.chat_stream(content):
                full_response += token
                await websocket.send_json({"type": "token", "content": token})

            await websocket.send_json(
                {"type": "done", "full_response": full_response}
            )
    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté")
    except Exception:
        logger.exception("Erreur WebSocket")
        try:
            await websocket.send_json(
                {"type": "error", "content": "Erreur interne du serveur"}
            )
        except Exception:
            pass
