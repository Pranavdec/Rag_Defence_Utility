from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class RetrievalCache:
    """
    Tiny on-disk cache for retrieval results.

    Stores the *formatted* VectorStore.query outputs (list[dict]) keyed by:
      - collection_name (dataset)
      - embedding_model (to avoid cross-embedder mismatches)
      - query_text
      - top_k
      - include_embeddings
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        query_text: str,
        top_k: int,
        include_embeddings: bool,
    ) -> str:
        blob = json.dumps(
            {
                "collection": collection_name,
                "embedding_model": embedding_model,
                "query_text": query_text,
                "top_k": int(top_k),
                "include_embeddings": bool(include_embeddings),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return _sha1(blob)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        query_text: str,
        top_k: int,
        include_embeddings: bool,
    ) -> Optional[List[Dict[str, Any]]]:
        key = self._key(
            collection_name=collection_name,
            embedding_model=embedding_model,
            query_text=query_text,
            top_k=top_k,
            include_embeddings=include_embeddings,
        )
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
        return None

    def put(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        query_text: str,
        top_k: int,
        include_embeddings: bool,
        results: List[Dict[str, Any]],
    ) -> None:
        key = self._key(
            collection_name=collection_name,
            embedding_model=embedding_model,
            query_text=query_text,
            top_k=top_k,
            include_embeddings=include_embeddings,
        )
        path = self._path(key)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

