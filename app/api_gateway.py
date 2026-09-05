from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.auth import require_scope, RateLimitMiddleware, ApiClient
from app.document_pipeline import LegalDocumentIntelligenceHub
from app.clause_extraction_engine import ClauseExtractionEngine
from app.risk_intelligence import ContractRiskIntelligence
from app.obligation_tracking import ObligationTrackingEngine
from app.knowledge_graph import LegalKnowledgeGraph
from app.compliance_engine import ComplianceIntelligenceEngine
from app.alert_center import LegalAlertCenter
from app.legal_copilot import AILegalCopilot
from app.database import init_db
from app.cache import legal_cache
from app.audit import AuditLoggingMiddleware, get_recent_audit_entries, get_audit_summary
from app.bonus_contract_negotiation import AIContractNegotiationAssistant
from app.event_bus import EventDrivenDocumentProcessor, legal_event_bus, EventType


app = FastAPI(
    title="Enterprise Legal Intelligence & Contract Reasoning Platform",
    version="1.0.0",
    description="AI-234 : Legal Document Intelligence, Clause Extraction, Risk Intelligence, Obligation Tracking, Knowledge Graph, and AI Legal Copilot",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=300)
app.add_middleware(AuditLoggingMiddleware)

document_hub = LegalDocumentIntelligenceHub()
clause_engine = ClauseExtractionEngine()
risk_engine = ContractRiskIntelligence()
obligation_engine = ObligationTrackingEngine()
knowledge_graph = LegalKnowledgeGraph()
compliance_engine = ComplianceIntelligenceEngine()
alert_center = LegalAlertCenter()
copilot = AILegalCopilot()
negotiation_assistant = AIContractNegotiationAssistant()
event_processor = EventDrivenDocumentProcessor()


class CopilotQuery(BaseModel):
    question: str


class ClauseAnalysisRequest(BaseModel):
    document_code: str
    document_text: str


class ComparisonRequest(BaseModel):
    text_a: str
    text_b: str


@app.on_event("startup")
def on_startup() -> None:
    try:
        init_db()
    except Exception:
        pass


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "environment": settings.service.environment,
    }


@app.get("/dashboard")
def serve_dashboard() -> FileResponse:
    dashboard_path = Path(__file__).parent / "dashboard.html"
    return FileResponse(dashboard_path, media_type="text/html")


@app.post("/documents/ingest")
def ingest_documents(count: int = 200, client: ApiClient = Depends(require_scope("write"))) -> dict:
    result = document_hub.ingest_batch(count)
    return {k: v for k, v in result.items() if k != "documents"}


@app.post("/clauses/analyze")
def analyze_clauses(payload: ClauseAnalysisRequest, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return clause_engine.analyze_document(payload.document_code, payload.document_text)


@app.get("/risk/top-contracts")
def top_risk_contracts(limit: int = 10, client: ApiClient = Depends(require_scope("read"))) -> dict:
    assessments = risk_engine.top_risk_contracts(limit)
    return {
        "contracts": [
            {
                "document_code": a.document_code,
                "risk_level": a.risk_level,
                "risk_score": a.risk_score,
                "risk_factors": a.risk_factors,
                "financial_exposure": a.financial_exposure,
            }
            for a in assessments
        ]
    }


@app.get("/obligations/tracking")
def obligation_tracking(sample_size: int = 150, client: ApiClient = Depends(require_scope("read"))) -> dict:
    cache_key = legal_cache._make_key("obligation_tracking", sample_size)
    cached = legal_cache.get(cache_key)
    if cached is not None:
        return cached
    result = obligation_engine.track_portfolio(sample_size)
    legal_cache.set(cache_key, result, ttl_seconds=180)
    return result


@app.post("/knowledge-graph/seed")
def seed_knowledge_graph(num_contracts: int = 50, client: ApiClient = Depends(require_scope("write"))) -> dict:
    return knowledge_graph.seed_sample_graph(num_contracts)


@app.get("/knowledge-graph/summary")
def knowledge_graph_summary(client: ApiClient = Depends(require_scope("read"))) -> dict:
    return knowledge_graph.summary()


@app.get("/knowledge-graph/vendor-exposure")
def vendor_exposure(top_k: int = 10, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return {"vendors": knowledge_graph.find_high_exposure_vendors(top_k)}


@app.get("/knowledge-graph/criticality")
def contract_criticality(client: ApiClient = Depends(require_scope("read"))) -> dict:
    return {"ranking": knowledge_graph.contract_criticality_ranking()}


@app.get("/compliance/portfolio")
def compliance_portfolio(sample_size: int = 100, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return compliance_engine.evaluate_portfolio(sample_size)


@app.get("/compliance/regulatory-changes")
def regulatory_changes(client: ApiClient = Depends(require_scope("read"))) -> dict:
    return {"changes": compliance_engine.get_regulatory_changes()}


@app.post("/contracts/compare")
def compare_contracts(payload: ComparisonRequest, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return copilot.executor.comparison_engine.compare(payload.text_a, payload.text_b)


@app.get("/alerts/sweep")
def alerts_sweep(client: ApiClient = Depends(require_scope("read"))) -> dict:
    return alert_center.run_full_sweep()


@app.post("/copilot/ask")
def copilot_ask(payload: CopilotQuery, client: ApiClient = Depends(require_scope("read"))) -> dict:
    response = copilot.ask(payload.question)
    return {
        "answer": response.answer,
        "tool_calls": response.tool_calls,
        "agents_consulted": response.agents_consulted,
        "generated_at": response.generated_at.isoformat(),
    }


@app.get("/copilot/knowledge-search")
def knowledge_search(query: str, top_k: int = 3, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return {"query": query, "results": copilot.executor.vector_store.search(query, top_k)}


@app.get("/audit/recent")
def audit_recent(limit: int = 100, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return {"entries": get_recent_audit_entries(limit)}


@app.get("/audit/summary")
def audit_summary(hours: int = 24, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return get_audit_summary(hours)


@app.post("/events/ingest-document")
def event_driven_ingest(payload: ClauseAnalysisRequest, client: ApiClient = Depends(require_scope("write"))) -> dict:
    return event_processor.ingest_and_process(payload.document_code, payload.document_text)


@app.get("/events/stream")
def event_stream(event_type: Optional[str] = None, limit: int = 50, client: ApiClient = Depends(require_scope("read"))) -> dict:
    filter_type = EventType(event_type) if event_type else None
    return {"events": legal_event_bus.recent_events(filter_type, limit)}


@app.get("/events/summary")
def event_summary(client: ApiClient = Depends(require_scope("read"))) -> dict:
    return legal_event_bus.stream_summary()


@app.post("/bonus/negotiation-redlines")
def negotiation_redlines(payload: ClauseAnalysisRequest, client: ApiClient = Depends(require_scope("read"))) -> dict:
    return negotiation_assistant.generate_redlines(payload.document_code, payload.document_text)


@app.get("/dashboard/executive-summary")
def executive_summary(client: ApiClient = Depends(require_scope("read"))) -> dict:
    cache_key = legal_cache._make_key("executive_summary")
    cached = legal_cache.get(cache_key)
    if cached is not None:
        return cached

    top_risk = risk_engine.top_risk_contracts(limit=5)
    tracking = obligation_engine.track_portfolio(sample_size=150)
    compliance = compliance_engine.evaluate_portfolio(sample_size=80)
    alerts = alert_center.run_full_sweep()

    avg_risk = sum(a.risk_score for a in top_risk) / len(top_risk) if top_risk else 0.0
    legal_health_score = max(0.0, 100 - avg_risk * 40 - (100 - compliance["compliance_rate_pct"]) * 0.3 - alerts["total_alerts"] * 0.5)

    payload = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "legal_health_score": round(legal_health_score, 1),
        "compliance_rate_pct": compliance["compliance_rate_pct"],
        "expiring_contracts_60_days": tracking["expiring_contracts_60_days"],
        "overdue_obligations": tracking["overdue_count"],
        "top_risk_contracts": [
            {"document_code": a.document_code, "risk_level": a.risk_level, "risk_score": a.risk_score}
            for a in top_risk
        ],
        "active_alerts": alerts["total_alerts"],
        "severity_breakdown": alerts["severity_breakdown"],
    }
    legal_cache.set(cache_key, payload, ttl_seconds=120)
    return payload
