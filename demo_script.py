from __future__ import annotations

import os
import sys
import json

import requests

BASE_URL = os.getenv("LEGALINTEL_API_URL", "http://localhost:8000")
API_KEY = os.getenv("LEGALINTEL_DEMO_API_KEY", "demo-key-change-me")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _call(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE_URL}{path}"
    response = requests.request(method, url, headers=HEADERS, timeout=30, **kwargs)
    print(f"[{response.status_code}] {method} {path}")
    if response.status_code >= 400:
        print(f"  -> {response.text[:300]}")
        return {}
    return response.json()


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_demo() -> None:
    _section("1. Health Check")
    _call("GET", "/health")

    _section("2. Legal Document Intelligence - Ingest")
    ingest = _call("POST", "/documents/ingest", params={"count": 100})
    print(json.dumps(ingest, indent=2)[:500])

    _section("3. Clause Extraction Engine")
    sample_text = (
        "Payment shall be made within 30 days of invoice receipt. "
        "The party shall have unlimited liability with no cap on damages. "
        "This agreement is governed by the laws of Singapore."
    )
    analysis = _call("POST", "/clauses/analyze", data=json.dumps({"document_code": "DEMO-001", "document_text": sample_text}))
    print(f"  Clauses found: {analysis.get('clauses_found')}, types: {analysis.get('clause_types_present')}")

    _section("4. Contract Risk Intelligence")
    risk = _call("GET", "/risk/top-contracts", params={"limit": 5})
    for c in risk.get("contracts", [])[:3]:
        print(f"  {c['document_code']}: {c['risk_score']:.0%} ({c['risk_level']})")

    _section("5. Obligation Tracking Engine")
    tracking = _call("GET", "/obligations/tracking", params={"sample_size": 100})
    print(f"  Total obligations: {tracking.get('total_obligations')}, expiring contracts: {tracking.get('expiring_contracts_60_days')}")

    _section("6. Legal Knowledge Graph")
    _call("POST", "/knowledge-graph/seed", params={"num_contracts": 30})
    kg_summary = _call("GET", "/knowledge-graph/summary")
    print(json.dumps(kg_summary, indent=2))

    _section("7. Compliance Intelligence")
    compliance = _call("GET", "/compliance/portfolio", params={"sample_size": 60})
    print(f"  Compliance rate: {compliance.get('compliance_rate_pct')}%")

    _section("8. AI Legal Copilot")
    copilot_response = _call(
        "POST", "/copilot/ask", data=json.dumps({"question": "Which contracts violate company policy?"})
    )
    print(f"  Copilot: {copilot_response.get('answer')}")

    _section("9. Bonus Challenge - AI Contract Negotiation Assistant")
    redlines = _call(
        "POST", "/bonus/negotiation-redlines",
        data=json.dumps({"document_code": "DEMO-001", "document_text": sample_text})
    )
    print(f"  Redlines suggested: {redlines.get('redline_count')}")

    _section("10. Alert Center")
    alerts = _call("GET", "/alerts/sweep")
    print(f"  Total alerts: {alerts.get('total_alerts')}, breakdown: {alerts.get('severity_breakdown')}")

    _section("11. Executive Dashboard Summary")
    summary = _call("GET", "/dashboard/executive-summary")
    print(json.dumps(summary, indent=2))

    _section("12. Audit Trail")
    audit_summary = _call("GET", "/audit/summary", params={"hours": 1})
    print(json.dumps(audit_summary, indent=2))

    _section("Demo Complete")
    print("All 9 core modules + Alert Center + Bonus Challenge exercised successfully.")


if __name__ == "__main__":
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"Cannot reach API at {BASE_URL}. Start it first with:")
        print("  uvicorn app.api_gateway:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    run_demo()
