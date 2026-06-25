"""Mémoire conversationnelle court terme."""

import threading
from datetime import datetime
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from config.settings import settings

_lock = threading.Lock()
_history_store: dict[str, InMemoryChatMessageHistory] = {}


class ShortTermMemory:
    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.max_messages = settings.max_conversation_history * 2

    def _get_history(self) -> InMemoryChatMessageHistory:
        with _lock:
            if self.session_id not in _history_store:
                _history_store[self.session_id] = InMemoryChatMessageHistory()
            return _history_store[self.session_id]

    def add_user_message(self, content: str) -> None:
        with _lock:
            history = self._get_history()
            history.add_user_message(content)
            self._trim(history)

    def add_ai_message(self, content: str) -> None:
        with _lock:
            history = self._get_history()
            history.add_ai_message(content)
            self._trim(history)

    def _trim(self, history: InMemoryChatMessageHistory) -> None:
        messages = history.messages
        if len(messages) > self.max_messages:
            history.messages = messages[-self.max_messages :]

    def get_messages(self) -> list[dict[str, Any]]:
        with _lock:
            history = self._get_history()
            result = []
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                else:
                    role = "system"
                result.append(
                    {
                        "role": role,
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            return result

    def get_context_string(self) -> str:
        with _lock:
            history = self._get_history()
            lines = []
            for msg in history.messages[-self.max_messages :]:
                role = (
                    "Utilisateur"
                    if isinstance(msg, HumanMessage)
                    else settings.jarvis_name
                )
                lines.append(f"{role}: {msg.content}")
            return "\n".join(lines)

    def clear(self) -> None:
        with _lock:
            if self.session_id in _history_store:
                del _history_store[self.session_id]
            logger.info(f"Historique effacé pour session {self.session_id}")

    @staticmethod
    def clear_all() -> None:
        with _lock:
            _history_store.clear()
            logger.info("Tout l'historique conversationnel a été effacé")
