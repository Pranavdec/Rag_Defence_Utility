from __future__ import annotations

import os
from typing import Any, Dict, Optional


def is_offline(config: Dict[str, Any]) -> bool:
    return bool(config.get("system", {}).get("offline", False))


def apply_offline_env(config: Dict[str, Any]) -> None:
    """
    Apply environment variables for HuggingFace offline mode.

    Once artifacts are cached, this prevents any network calls.
    """
    if not is_offline(config):
        return

    paths = config.get("paths", {}) or {}
    cache_root = str(paths.get("cache", "data/raw"))
    hf_home = str(paths.get("hf_home", os.path.join(cache_root, "hf_home")))

    os.makedirs(hf_home, exist_ok=True)

    # HuggingFace hub + transformers + datasets offline switches
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

