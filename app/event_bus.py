from __future__ import annotations

import json
import enum
import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

from app.config import settings


class EventType(str, enum.Enum):
    DOCUMENT_INGESTED = "document.ingested"
    CLAUSES_EXTRACTED = "clauses.extracted"
    RISK_ASSESSED = "risk.assessed"
    COMPLIANCE_EVALUATED = "compliance.evaluated"
    OBLIGATION_DUE_SOON = "obligation.due_soon"
    ALERT_RAISED = "alert.raised"


@dataclass
class Event:
    event_type: EventType
    payload: dict
    emitted_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())
    source: str = "system"

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type.value,
                "payload": self.payload,
                "emitted_at": self.emitted_at,
                "source": self.source,
            }
        )

    @staticmethod
    def from_json(raw: str) -> "Event":
        data = json.loads(raw)
        return Event(
            event_type=EventType(data["event_type"]),
            payload=data["payload"],
            emitted_at=data["emitted_at"],
            source=data.get("source", "system"),
        )


class RedisEventBackend:
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
                socket_connect_timeout=2,
                decode_responses=True,
            )
            self._client.ping()
        except Exception:
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def publish(self, channel: str, event: Event) -> bool:
        if not self._client:
            return False
        self._client.publish(channel, event.to_json())
        self._client.lpush(f"stream:{channel}", event.to_json())
        self._client.ltrim(f"stream:{channel}", 0, 499)
        return True

    def recent_events(self, channel: str, limit: int = 50) -> list[Event]:
        if not self._client:
            return []
        raw_events = self._client.lrange(f"stream:{channel}", 0, limit - 1)
        return [Event.from_json(raw) for raw in raw_events]


class LegalEventBus:
    def __init__(self) -> None:
        self.backend = RedisEventBackend()
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}
        self._local_log: list[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: EventType, payload: dict, source: str = "system") -> Event:
        event = Event(event_type=event_type, payload=payload, source=source)

        self.backend.publish(f"legalintel.{event_type.value}", event)
        self._local_log.append(event)
        if len(self._local_log) > 500:
            self._local_log = self._local_log[-500:]

        for handler in self._subscribers.get(event_type, []):
            try:
                handler(event)
            except Exception:
                pass

        return event

    def recent_events(self, event_type: EventType | None = None, limit: int = 50) -> list[dict]:
        if self.backend.is_connected and event_type:
            events = self.backend.recent_events(f"legalintel.{event_type.value}", limit)
        else:
            events = self._local_log[-limit:]
            if event_type:
                events = [e for e in events if e.event_type == event_type]

        return [
            {
                "event_type": e.event_type.value,
                "payload": e.payload,
                "emitted_at": e.emitted_at,
                "source": e.source,
            }
            for e in reversed(events)
        ]

    def stream_summary(self) -> dict:
        counts: dict[str, int] = {}
        for e in self._local_log:
            counts[e.event_type.value] = counts.get(e.event_type.value, 0) + 1
        return {
            "backend": "redis_pubsub" if self.backend.is_connected else "in_process_fallback",
            "total_events_this_session": len(self._local_log),
            "event_type_breakdown": counts,
        }


legal_event_bus = LegalEventBus()


class EventDrivenDocumentProcessor:
    def __init__(self, event_bus: LegalEventBus | None = None) -> None:
        self.event_bus = event_bus or legal_event_bus
        self._register_reactive_handlers()

    def _register_reactive_handlers(self) -> None:
        self.event_bus.subscribe(EventType.DOCUMENT_INGESTED, self._on_document_ingested)
        self.event_bus.subscribe(EventType.CLAUSES_EXTRACTED, self._on_clauses_extracted)
        self.event_bus.subscribe(EventType.RISK_ASSESSED, self._on_risk_assessed)

    def _on_document_ingested(self, event: Event) -> None:
        from app.clause_extraction_engine import ClauseExtractionEngine

        engine = ClauseExtractionEngine()
        document_code = event.payload.get("document_code")
        document_text = event.payload.get("document_text", "")
        analysis = engine.analyze_document(document_code, document_text)

        self.event_bus.publish(
            EventType.CLAUSES_EXTRACTED,
            {
                "document_code": document_code,
                "clauses_found": analysis["clauses_found"],
                "missing_clause_types": analysis["missing_clause_types"],
            },
            source="clause_extraction_engine",
        )

    def _on_clauses_extracted(self, event: Event) -> None:
        if not event.payload.get("missing_clause_types"):
            return
        self.event_bus.publish(
            EventType.ALERT_RAISED,
            {
                "document_code": event.payload.get("document_code"),
                "reason": "missing_clauses",
                "missing_clause_types": event.payload.get("missing_clause_types"),
            },
            source="event_driven_processor",
        )

    def _on_risk_assessed(self, event: Event) -> None:
        if event.payload.get("risk_level") not in {"high", "critical"}:
            return
        self.event_bus.publish(
            EventType.ALERT_RAISED,
            {
                "document_code": event.payload.get("document_code"),
                "reason": "high_risk_score",
                "risk_level": event.payload.get("risk_level"),
            },
            source="event_driven_processor",
        )

    def ingest_and_process(self, document_code: str, document_text: str, contract_value: float = 0.0) -> dict:
        ingest_event = self.event_bus.publish(
            EventType.DOCUMENT_INGESTED,
            {"document_code": document_code, "document_text": document_text, "contract_value": contract_value},
            source="document_pipeline",
        )
        return {
            "document_code": document_code,
            "ingest_event_emitted_at": ingest_event.emitted_at,
            "processing": "reactive_pipeline_triggered",
        }
