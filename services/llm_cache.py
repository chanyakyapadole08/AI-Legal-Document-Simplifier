"""
Tiny disk cache for Gemini results.

Same clause + same model => same answer, so a clause that was
already answered by Gemini never costs another API request.
Only real Gemini results are stored here (never rule/Qwen
fallback output), so a later run with quota available can still
upgrade those clauses to Gemini quality.

Delete cache/llm_cache.json to reset.
"""

import hashlib
import json
import os
import threading

CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")

_lock = threading.Lock()
_data = None


def make_key(*parts) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load() -> dict:
    global _data

    if _data is not None:
        return _data

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        _data = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _data = {}

    return _data


def _save(data: dict) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp_path = CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, CACHE_FILE)
    except Exception as e:
        print(f"[CACHE] Write failed: {e}")


def get(key: str):
    with _lock:
        return _load().get(key)


def put_many(pairs: list) -> None:
    """pairs: list of (key, value)."""
    if not pairs:
        return

    with _lock:
        data = _load()
        for key, value in pairs:
            data[key] = value
        _save(data)