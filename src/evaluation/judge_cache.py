from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class JudgeCache:
    """Content-addressed cache for per-sample judge metric scores."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return _sha1(blob)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, payload: Dict[str, Any]) -> Optional[float]:
        key = self._key(payload)
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "score" in obj:
                v = obj["score"]
                if isinstance(v, (int, float)):
                    return float(v)
        except Exception:
            return None
        return None

    def put(self, payload: Dict[str, Any], score: float) -> None:
        key = self._key(payload)
        path = self._path(key)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"score": float(score)}, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

