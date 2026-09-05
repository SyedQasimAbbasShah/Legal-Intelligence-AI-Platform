from __future__ import annotations

import time
import datetime as dt

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.database import get_session, AuditLog
from app.auth import get_client_for_key


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        raw_key = request.headers.get("X-API-Key")
        client = get_client_for_key(raw_key)
        actor = client.client_id if client else "anonymous"

        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 2)

        try:
            with get_session() as session:
                session.add(
                    AuditLog(
                        actor=actor,
                        action=f"{request.method} {request.url.path}",
                        entity=request.url.path.strip("/").split("/")[0] if request.url.path != "/" else "root",
                        timestamp=dt.datetime.now(dt.UTC),
                        details={
                            "status_code": response.status_code,
                            "duration_ms": duration_ms,
                            "query_params": dict(request.query_params),
                        },
                    )
                )
        except Exception:
            pass

        return response


def get_recent_audit_entries(limit: int = 100) -> list[dict]:
    try:
        with get_session() as session:
            entries = (
                session.query(AuditLog)
                .order_by(AuditLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "actor": e.actor,
                    "action": e.action,
                    "entity": e.entity,
                    "timestamp": e.timestamp.isoformat(),
                    "details": e.details,
                }
                for e in entries
            ]
    except Exception:
        return []


def get_audit_summary(hours: int = 24) -> dict:
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
    try:
        with get_session() as session:
            entries = session.query(AuditLog).filter(AuditLog.timestamp >= cutoff).all()
    except Exception:
        return {
            "window_hours": hours,
            "total_requests": 0,
            "by_action": {},
            "by_actor": {},
            "note": "Audit database unavailable; returning empty summary.",
        }

    action_counts: dict[str, int] = {}
    actor_counts: dict[str, int] = {}
    for e in entries:
        action_counts[e.action] = action_counts.get(e.action, 0) + 1
        actor_counts[e.actor] = actor_counts.get(e.actor, 0) + 1

    return {
        "window_hours": hours,
        "total_requests": len(entries),
        "by_action": action_counts,
        "by_actor": actor_counts,
    }
