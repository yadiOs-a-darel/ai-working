"""Mémoire long terme vectorielle avec ChromaDB."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config.settings import settings

_lock = threading.Lock()
_long_term_instance: Optional["LongTermMemory"] = None


class LongTermMemory:
    COLLECTION_NAME = "jarvis_memory"

    def __init__(self) -> None:
        self.persist_dir = Path(settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB initialisé ({self._collection.count()} souvenirs)"
            )
        except Exception:
            logger.exception("Erreur initialisation ChromaDB")
            raise

    def add(self, content: str, metadata: dict | None = None) -> None:
        if not content.strip():
            return
        with _lock:
            try:
                import uuid
                from datetime import datetime

                doc_id = str(uuid.uuid4())
                meta = metadata or {}
                meta["timestamp"] = datetime.now().isoformat()
                self._collection.add(
                    documents=[content],
                    metadatas=[meta],
                    ids=[doc_id],
                )
                logger.debug(f"Souvenir ajouté : {content[:60]}...")
            except Exception:
                logger.exception("Erreur ajout mémoire long terme")

    def search(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        with _lock:
            try:
                if self._collection.count() == 0:
                    return []
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count()),
                )
                items = []
                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        items.append(
                            {
                                "content": doc,
                                "metadata": (
                                    results["metadatas"][0][i]
                                    if results.get("metadatas")
                                    else {}
                                ),
                                "distance": (
                                    results["distances"][0][i]
                                    if results.get("distances")
                                    else None
                                ),
                            }
                        )
                return items
            except Exception:
                logger.exception("Erreur recherche mémoire")
                return []

    def get_relevant_context(self, query: str, timeout: float = 10.0) -> str:
        if not settings.memory_search_enabled:
            return ""

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.search, query, 3)
                memories = future.result(timeout=timeout)
        except FuturesTimeout:
            logger.warning("Recherche mémoire expirée — ignorée")
            return ""
        except Exception:
            logger.warning("Recherche mémoire échouée — ignorée")
            return ""

        if not memories:
            return ""
        lines = ["Souvenirs pertinents :"]
        for mem in memories:
            lines.append(f"- {mem['content']}")
        return "\n".join(lines)

    def count(self) -> int:
        with _lock:
            try:
                return self._collection.count()
            except Exception:
                return 0

    def save_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Sauvegarde un échange important."""
        content = f"Utilisateur: {user_msg}\n{settings.jarvis_name}: {assistant_msg}"

        def _save() -> None:
            self.add(content, metadata={"type": "conversation"})

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_save).result(timeout=10)
        except FuturesTimeout:
            logger.warning("Sauvegarde mémoire expirée — ignorée")
        except Exception:
            logger.warning("Sauvegarde mémoire échouée — ignorée")


def get_long_term_memory() -> LongTermMemory:
    global _long_term_instance
    if _long_term_instance is None:
        _long_term_instance = LongTermMemory()
    return _long_term_instance
