from __future__ import annotations

import json
import hashlib
import functools
from typing import Callable, Any

from app.config import settings


class LegalCache:
    def __init__(self) -> None:
        self._client = None
        try:
            import redis

            self._client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password or None,
                ssl=settings.redis.ssl,
                socket_connect_timeout=3,
                decode_responses=True,
            )
            self._client.ping()
        except Exception:
            self._client = None
        self._local_store: dict[str, str] = {}

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def _make_key(self, namespace: str, *args, **kwargs) -> str:
        raw = f"{namespace}:{args}:{sorted(kwargs.items())}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        return f"legalintel:{namespace}:{digest}"

    def get(self, key: str) -> Any | None:
        if self._client:
            value = self._client.get(key)
        else:
            value = self._local_store.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        serialized = json.dumps(value)
        if self._client:
            self._client.set(key, serialized, ex=ttl_seconds)
        else:
            self._local_store[key] = serialized

    def invalidate(self, key: str) -> None:
        if self._client:
            self._client.delete(key)
        else:
            self._local_store.pop(key, None)


legal_cache = LegalCache()
