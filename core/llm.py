"""Client LLM avec Ollama local et fallback OpenAI."""

import threading
from typing import Any, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from config.settings import settings

_llm_instance: Optional["LLMClient"] = None


class LLMClient:
    def __init__(self) -> None:
        self._ollama_available = self.is_ollama_available()
        self._active_provider = "ollama" if self._ollama_available else "openai"
        self._chat_model: Optional[BaseChatModel] = None
        self._tools_supported: Optional[bool] = None
        logger.info(
            f"LLM initialisé : {self._active_provider} "
            f"({settings.ollama_model if self._ollama_available else 'gpt-4o-mini'})"
        )

    def is_ollama_available(self) -> bool:
        """Vérifie si Ollama répond sur le endpoint local."""
        try:
            response = requests.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=3,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> dict[str, Any]:
        """Retourne le modèle actif et sa disponibilité."""
        if self._ollama_available:
            model = settings.ollama_model
            provider = "ollama"
        elif settings.openai_api_key:
            model = "gpt-4o-mini"
            provider = "openai"
        else:
            model = settings.ollama_fallback_model
            provider = "unavailable"

        return {
            "provider": provider,
            "model": model,
            "ollama_available": self._ollama_available,
            "ollama_url": settings.ollama_base_url,
        }

    def model_supports_tools(self) -> bool:
        """Indique si le modèle actif supporte le tool calling."""
        if self._tools_supported is not None:
            return self._tools_supported

        if self._ollama_available:
            try:
                response = requests.post(
                    f"{settings.ollama_base_url}/api/show",
                    json={"name": settings.ollama_model},
                    timeout=5,
                )
                if response.status_code == 200:
                    caps = response.json().get("capabilities", [])
                    self._tools_supported = "tools" in caps
                else:
                    self._tools_supported = False
            except Exception:
                self._tools_supported = False
        elif settings.openai_api_key:
            self._tools_supported = True
        else:
            self._tools_supported = False

        if not self._tools_supported:
            model = (
                settings.ollama_model
                if self._ollama_available
                else "gpt-4o-mini"
            )
            logger.warning(
                f"Modèle {model} sans support outils — mode chat simple"
            )

        return self._tools_supported

    def chat_local(
        self,
        messages: list[dict[str, str]],
        timeout: int | None = None,
    ) -> str:
        """Appel direct à Ollama local (évite LangChain et le cloud)."""
        if not self._ollama_available:
            raise RuntimeError("Ollama indisponible sur " + settings.ollama_base_url)

        timeout = timeout or settings.ollama_timeout
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": settings.ollama_num_ctx},
        }
        result: dict[str, str] = {}
        errors: list[Exception] = []

        def _request() -> None:
            try:
                response = requests.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json=payload,
                    timeout=(10, timeout + 30),
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
                if not content.strip():
                    raise RuntimeError("Réponse Ollama vide")
                result["content"] = content
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=_request, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Ollama n'a pas répondu en {timeout}s — "
                "essayez phi3.5 (plus rapide) ou augmentez OLLAMA_TIMEOUT."
            )

        if errors:
            exc = errors[0]
            if isinstance(exc, requests.Timeout):
                raise TimeoutError(
                    f"Ollama n'a pas répondu en {timeout}s — "
                    "modèle trop lourd pour le CPU."
                ) from exc
            raise RuntimeError(f"Erreur Ollama : {exc}") from exc

        return result["content"]

    @property
    def chat_model(self) -> BaseChatModel:
        """Retourne le BaseChatModel LangChain selon disponibilité."""
        if self._chat_model is not None:
            return self._chat_model

        if self._ollama_available:
            from langchain_ollama import ChatOllama

            self._chat_model = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.7,
                num_ctx=settings.ollama_num_ctx,
            )
        elif settings.openai_api_key:
            from langchain_openai import ChatOpenAI

            self._chat_model = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.openai_api_key,
                temperature=0.7,
            )
            self._active_provider = "openai"
        else:
            from langchain_ollama import ChatOllama

            logger.warning("Ollama et OpenAI indisponibles, tentative fallback Ollama")
            self._chat_model = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_fallback_model,
                temperature=0.7,
                num_ctx=settings.ollama_num_ctx,
            )

        return self._chat_model

    def refresh_availability(self) -> None:
        """Rafraîchit la disponibilité Ollama."""
        was_available = self._ollama_available
        self._ollama_available = self.is_ollama_available()
        if was_available != self._ollama_available:
            self._chat_model = None
            logger.info(f"Disponibilité Ollama changée : {self._ollama_available}")


def get_llm_client() -> LLMClient:
    """Singleton LLM client."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
