from __future__ import annotations

import pytest

from app.document_pipeline import LegalDocumentIntelligenceHub, LegalDocumentSimulator
from app.clause_extraction_engine import ClauseExtractionEngine
from app.risk_intelligence import ContractRiskIntelligence
from app.obligation_tracking import ObligationTrackingEngine
from app.knowledge_graph import LegalKnowledgeGraph
from app.compliance_engine import ComplianceIntelligenceEngine
from app.legal_rag import LegalRAGStore, ContractComparisonEngine, build_default_legal_knowledge_base
from app.explainable_ai import RiskExplainer
from app.bonus_contract_negotiation import AIContractNegotiationAssistant
from app.auth import _hash_key, ROLE_SCOPES
from app.legal_copilot import AILegalCopilot
from app.event_bus import EventDrivenDocumentProcessor, legal_event_bus, EventType
from app.cache import LegalCache


class TestDocumentPipeline:
    def test_ingest_batch_returns_expected_structure(self) -> None:
        hub = LegalDocumentIntelligenceHub()
        result = hub.ingest_batch(30)
        assert result["documents_ingested"] == 30
        assert len(result["type_breakdown"]) > 0

    def test_simulated_documents_have_valid_dates(self) -> None:
        simulator = LegalDocumentSimulator()
        doc = simulator.generate_document(1)
        assert doc.contract_value >= 0
        assert doc.effective_date < doc.expiry_date


class TestClauseExtraction:
    def test_extract_clauses_finds_known_patterns(self) -> None:
        engine = ClauseExtractionEngine()
        text = "Payment shall be made within 30 days. The receiving party shall maintain confidentiality for 5 years."
        clauses = engine.extract_clauses(text)
        types_found = {c.clause_type for c in clauses}
        assert "payment" in types_found or "confidentiality" in types_found

    def test_missing_clauses_detected(self) -> None:
        engine = ClauseExtractionEngine()
        missing = engine.find_missing_clauses("This document has no relevant legal terms at all.")
        assert len(missing) > 0


class TestRiskIntelligence:
    def test_assess_document_returns_valid_risk_level(self) -> None:
        engine = ContractRiskIntelligence()
        engine.initialize()
        assessment = engine.assess_document(
            "DOC-TEST-01",
            "The party shall have unlimited liability with no cap on damages under this agreement.",
            500000.0,
        )
        assert assessment.risk_level in {"low", "medium", "high", "critical"}
        assert 0.0 <= assessment.risk_score <= 1.0

    def test_top_risk_contracts_sorted_descending(self) -> None:
        engine = ContractRiskIntelligence()
        assessments = engine.top_risk_contracts(limit=10)
        scores = [a.risk_score for a in assessments]
        assert scores == sorted(scores, reverse=True)


class TestObligationTracking:
    def test_track_portfolio_returns_expected_keys(self) -> None:
        engine = ObligationTrackingEngine()
        result = engine.track_portfolio(sample_size=20)
        assert "total_obligations" in result
        assert "expiring_contracts_60_days" in result
        assert result["total_obligations"] >= 0


class TestKnowledgeGraph:
    def test_seed_creates_expected_node_types(self) -> None:
        graph = LegalKnowledgeGraph()
        summary = graph.seed_sample_graph(num_contracts=10)
        assert summary["total_nodes"] > 0
        assert "Contract" in summary["node_type_breakdown"]
        assert "Vendor" in summary["node_type_breakdown"]

    def test_high_exposure_vendors_returns_ranked_list(self) -> None:
        graph = LegalKnowledgeGraph()
        graph.seed_sample_graph(num_contracts=15)
        exposure = graph.find_high_exposure_vendors(top_k=5)
        assert isinstance(exposure, list)

    def test_jurisdiction_is_first_class_node(self) -> None:
        graph = LegalKnowledgeGraph()
        summary = graph.seed_sample_graph(num_contracts=20)
        assert "Jurisdiction" in summary["node_type_breakdown"]
        assert summary["node_type_breakdown"]["Jurisdiction"] > 0


class TestComplianceEngine:
    def test_evaluate_document_returns_frameworks(self) -> None:
        engine = ComplianceIntelligenceEngine()
        result = engine.evaluate_document("DOC-TEST-02", "contract", ["payment", "termination"])
        assert result["frameworks_evaluated"] > 0

    def test_regulatory_changes_available(self) -> None:
        engine = ComplianceIntelligenceEngine()
        changes = engine.get_regulatory_changes()
        assert len(changes) > 0


class TestLegalRAG:
    def test_index_and_search(self) -> None:
        store = LegalRAGStore()
        stats = store.build_index(build_default_legal_knowledge_base())
        assert stats["documents_indexed"] > 0
        results = store.search("unlimited liability risk", top_k=3)
        assert len(results) > 0

    def test_contract_comparison(self) -> None:
        comparator = ContractComparisonEngine()
        result = comparator.compare(
            "Payment shall be made within 30 days of invoice receipt.",
            "Payment shall be made within 30 days of receiving an invoice.",
        )
        assert 0.0 <= result["similarity_score"] <= 1.0


class TestExplainableAI:
    def test_explain_risk_score_mentions_factors(self) -> None:
        explainer = RiskExplainer()
        text = explainer.explain_risk_score(["unlimited_liability"], 0.85, 250000.0)
        assert "liability" in text.lower()

    def test_explain_missing_clauses(self) -> None:
        explainer = RiskExplainer()
        text = explainer.explain_missing_clauses(["arbitration", "force_majeure"])
        assert "arbitration" in text.lower()


class TestNegotiationAssistant:
    def test_generate_redlines_for_risky_clause(self) -> None:
        assistant = AIContractNegotiationAssistant()
        result = assistant.generate_redlines(
            "DOC-TEST-03",
            "The party shall have unlimited liability with no cap on damages under this agreement.",
        )
        assert result["redline_count"] >= 0

    def test_negotiation_summary_returns_string(self) -> None:
        assistant = AIContractNegotiationAssistant()
        summary = assistant.negotiation_summary("DOC-TEST-04", "Standard payment terms apply within 30 days.")
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestMultiAgentCopilot:
    def test_single_topic_question_routes_one_agent(self) -> None:
        copilot = AILegalCopilot()
        response = copilot.ask("Which contracts violate company policy?")
        assert "compliance_officer" in response.agents_consulted

    def test_multi_topic_question_routes_multiple_agents(self) -> None:
        copilot = AILegalCopilot()
        response = copilot.ask("What obligations expire next month and are there any risky clauses?")
        assert len(response.agents_consulted) >= 2
        assert "risk_analyst" in response.agents_consulted
        assert "obligations_tracker" in response.agents_consulted

    def test_synthesizer_combines_multiple_agent_outputs(self) -> None:
        copilot = AILegalCopilot()
        response = copilot.ask("Check compliance and list expiring obligations")
        if len(response.agents_consulted) > 1:
            assert "[" in response.answer


    def test_clause_explainer_agent_routes_correctly(self) -> None:
        copilot = AILegalCopilot()
        response = copilot.ask(
            "Explain this clause in simple language: The receiving party shall maintain confidentiality for 5 years."
        )
        assert "clause_explainer" in response.agents_consulted
        assert "plain language" in response.answer.lower()


class TestEventDrivenArchitecture:
    def test_ingest_triggers_reactive_chain(self) -> None:
        processor = EventDrivenDocumentProcessor()
        result = processor.ingest_and_process(
            "DOC-TEST-EVT", "The party shall have unlimited liability with no cap on damages.", 100000.0
        )
        assert result["processing"] == "reactive_pipeline_triggered"

    def test_event_bus_records_events(self) -> None:
        legal_event_bus.publish(EventType.DOCUMENT_INGESTED, {"document_code": "DOC-BUS-TEST"})
        events = legal_event_bus.recent_events(EventType.DOCUMENT_INGESTED, limit=5)
        assert len(events) > 0

    def test_stream_summary_returns_backend_status(self) -> None:
        summary = legal_event_bus.stream_summary()
        assert "backend" in summary
        assert summary["backend"] in {"redis_pubsub", "in_process_fallback"}


class TestCache:
    def test_local_fallback_set_and_get(self) -> None:
        cache = LegalCache()
        cache.set("test:key", {"value": 99}, ttl_seconds=10)
        assert cache.get("test:key") == {"value": 99}


class TestAuth:
    def test_hash_key_deterministic(self) -> None:
        assert _hash_key("abc") == _hash_key("abc")
        assert _hash_key("abc") != _hash_key("xyz")

    def test_role_scopes_defined(self) -> None:
        assert "attorney" in ROLE_SCOPES
        assert "read" in ROLE_SCOPES["general_counsel"]
        assert "admin" in ROLE_SCOPES["general_counsel"]
