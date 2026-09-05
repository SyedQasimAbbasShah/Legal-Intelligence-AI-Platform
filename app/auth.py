from __future__ import annotations

import os
import time
import hashlib
import secrets
from dataclasses import dataclass

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class ApiClient:
    client_id: str
    role: str
    key_hash: str


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _load_clients() -> dict[str, ApiClient]:
    raw = os.getenv("LEGALINTEL_API_KEYS", "")
    clients: dict[str, ApiClient] = {}
    if not raw:
        default_key = os.getenv("LEGALINTEL_DEFAULT_API_KEY", secrets.token_urlsafe(24))
        clients[_hash_key(default_key)] = ApiClient("default-user", "attorney", _hash_key(default_key))
        print(f"[legalintel.auth] No LEGALINTEL_API_KEYS set. Generated dev key: {default_key}")
        return clients

    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            continue
        client_id, role, raw_key = parts
        key_hash = _hash_key(raw_key)
        clients[key_hash] = ApiClient(client_id, role, key_hash)
    return clients


_CLIENTS = _load_clients()

ROLE_SCOPES = {
    "paralegal": {"read"},
    "attorney": {"read", "write"},
    "general_counsel": {"read", "write", "admin"},
}


def get_client_for_key(raw_key: str | None) -> ApiClient | None:
    if not raw_key:
        return None
    return _CLIENTS.get(_hash_key(raw_key))


def require_scope(scope: str):
    async def dependency(api_key: str | None = Security(API_KEY_HEADER)) -> ApiClient:
        client = get_client_for_key(api_key)
        if client is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        allowed = ROLE_SCOPES.get(client.role, set())
        if scope not in allowed:
            raise HTTPException(status_code=403, detail=f"Role '{client.role}' lacks '{scope}' scope")
        return client

    return dependency


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 300) -> None:
        super().__init__(app)
        self._capacity = requests_per_minute
        self._refill_rate = requests_per_minute / 60.0
        self._buckets: dict[str, _Bucket] = {}

    async def dispatch(self, request: Request, call_next):
        identity = request.headers.get("X-API-Key") or (request.client.host if request.client else "anonymous")
        now = time.monotonic()
        bucket = self._buckets.get(identity)
        if bucket is None:
            bucket = _Bucket(tokens=float(self._capacity), last_refill=now)
            self._buckets[identity] = bucket

        elapsed = now - bucket.last_refill
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
        bucket.last_refill = now

        if bucket.tokens < 1.0:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        bucket.tokens -= 1.0
        response = await call_next(request)
        return response
