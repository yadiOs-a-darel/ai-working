"""Recherche web Tavily + DuckDuckGo fallback."""

import requests
from loguru import logger

from config.settings import settings


def recherche_web(query: str) -> str:
    """Recherche des informations sur internet."""
    if not query.strip():
        return "Veuillez préciser votre recherche."

    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.tavily_api_key)
            results = client.search(query=query, max_results=3)
            formatted = []
            for item in results.get("results", [])[:3]:
                title = item.get("title", "Sans titre")
                content = item.get("content", "")[:300]
                url = item.get("url", "")
                formatted.append(f"Source: {title}\n{content}\nURL: {url}\n")
            if formatted:
                return "\n".join(formatted)
        except Exception:
            logger.exception("Erreur Tavily, fallback DuckDuckGo")

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"Aucun résultat trouvé pour : {query}"

        formatted = []
        for item in results[:3]:
            title = item.get("title", "Sans titre")
            body = item.get("body", "")[:300]
            url = item.get("href", "")
            formatted.append(f"Source: {title}\n{body}\nURL: {url}\n")
        return "\n".join(formatted)
    except Exception as exc:
        logger.exception("Erreur recherche web")
        return f"Recherche web impossible : {exc}"
