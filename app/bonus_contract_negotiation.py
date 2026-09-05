from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.clause_extraction_engine import ClauseExtractionEngine, CLAUSE_KEYWORDS


@dataclass
class RedlineSuggestion:
    clause_type: str
    original_text: str
    suggested_text: str
    rationale: str
    negotiation_leverage: str


PLAYBOOK = {
    "payment": {
        "risk_trigger": ["100%", "no cap", "unlimited"],
        "suggested_text": "Payment shall be made within 30 days of invoice receipt, with a late fee capped at 1.5% per month.",
        "rationale": "Uncapped or overly aggressive late fees create disproportionate financial risk relative to contract value.",
        "leverage": "Standard industry late fee terms are 1-2% monthly; cite market benchmarks to justify the counter.",
    },
    "termination": {
        "risk_trigger": ["sole discretion", "immediate", "no notice"],
        "suggested_text": "Either party may terminate this agreement with 30 days written notice for convenience.",
        "rationale": "Termination without adequate notice prevents operational continuity planning and increases transition risk.",
        "leverage": "Propose mutual notice periods to keep the clause balanced and defensible to both parties.",
    },
    "confidentiality": {
        "risk_trigger": ["perpetual", "indefinite"],
        "suggested_text": "The receiving party shall maintain confidentiality of disclosed information for a period of 5 years post-termination.",
        "rationale": "Perpetual confidentiality obligations are difficult to enforce and audit over time, and rarely reflect the true sensitivity window of the information.",
        "leverage": "Most enterprise NDAs cap confidentiality terms at 3-7 years; anchor to this range.",
    },
    "liability": {
        "risk_trigger": ["unlimited", "no cap", "uncapped"],
        "suggested_text": "Total liability under this agreement shall not exceed 100% of fees paid in the preceding 12 months.",
        "rationale": "Uncapped liability exposes the company to disproportionate financial risk relative to the contract's economic value.",
        "leverage": "A 12-month fees-paid cap is a widely accepted market standard and should be easy for counterparty to accept.",
    },
    "force_majeure": {
        "risk_trigger": ["narrow", "limited events"],
        "suggested_text": "Neither party shall be liable for delays caused by events beyond reasonable control, including natural disasters, government action, pandemics, and labor disputes.",
        "rationale": "A narrowly defined force majeure clause leaves the company exposed during disruptions not explicitly enumerated.",
        "leverage": "Broaden the enumerated list to include modern risk categories such as pandemics and cyber incidents.",
    },
    "renewal": {
        "risk_trigger": ["automatic", "perpetual renewal"],
        "suggested_text": "This agreement shall automatically renew for successive 1 year terms unless terminated with 60 days notice.",
        "rationale": "Long automatic renewal cycles without adequate opt-out notice can lock the company into unfavorable terms.",
        "leverage": "Shorten the renewal term and lengthen the notice window to preserve exit optionality.",
    },
    "governing_law": {
        "risk_trigger": ["foreign jurisdiction"],
        "suggested_text": "This agreement shall be governed by the laws of the company's home jurisdiction.",
        "rationale": "Foreign governing law increases litigation cost and unpredictability in the event of a dispute.",
        "leverage": "Propose home jurisdiction as governing law, or a neutral third jurisdiction as a compromise.",
    },
    "arbitration": {
        "risk_trigger": ["counterparty jurisdiction", "foreign arbitration"],
        "suggested_text": "Disputes shall be resolved through binding arbitration in a neutral jurisdiction agreed upon by both parties.",
        "rationale": "Arbitration in the counterparty's home jurisdiction disadvantages the company procedurally and increases travel and legal costs.",
        "leverage": "Propose a neutral arbitration venue such as Singapore or London for cross-border agreements.",
    },
}


class NegotiationRiskDetector:
    def detect_leverage_points(self, clause_type: str, clause_text: str) -> bool:
        playbook_entry = PLAYBOOK.get(clause_type)
        if not playbook_entry:
            return False
        lowered = clause_text.lower()
        return any(trigger in lowered for trigger in playbook_entry["risk_trigger"])


class AIContractNegotiationAssistant:
    def __init__(self) -> None:
        self.clause_engine = ClauseExtractionEngine()
        self.detector = NegotiationRiskDetector()

    def generate_redlines(self, document_code: str, document_text: str) -> dict:
        clauses = self.clause_engine.extract_clauses(document_text)
        suggestions = []

        for clause in clauses:
            playbook_entry = PLAYBOOK.get(clause.clause_type)
            if not playbook_entry:
                continue

            needs_redline = self.detector.detect_leverage_points(clause.clause_type, clause.clause_text)
            if not needs_redline and clause.confidence < 0.85:
                continue

            suggestions.append(
                RedlineSuggestion(
                    clause_type=clause.clause_type,
                    original_text=clause.clause_text,
                    suggested_text=playbook_entry["suggested_text"],
                    rationale=playbook_entry["rationale"],
                    negotiation_leverage=playbook_entry["leverage"],
                )
            )

        missing = self.clause_engine.find_missing_clauses(document_text)
        missing_suggestions = []
        for clause_type in missing:
            playbook_entry = PLAYBOOK.get(clause_type)
            if playbook_entry:
                missing_suggestions.append(
                    {
                        "clause_type": clause_type,
                        "recommendation": f"Add a standard {clause_type.replace('_', ' ')} clause",
                        "suggested_text": playbook_entry["suggested_text"],
                    }
                )

        return {
            "document_code": document_code,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "redline_count": len(suggestions),
            "missing_clause_recommendations": missing_suggestions,
            "redlines": [
                {
                    "clause_type": s.clause_type,
                    "original_text": s.original_text,
                    "suggested_text": s.suggested_text,
                    "rationale": s.rationale,
                    "negotiation_leverage": s.negotiation_leverage,
                }
                for s in suggestions
            ],
        }

    def negotiation_summary(self, document_code: str, document_text: str) -> str:
        result = self.generate_redlines(document_code, document_text)
        if result["redline_count"] == 0 and not result["missing_clause_recommendations"]:
            return f"{document_code} requires no immediate redlines; terms are within standard risk tolerance."

        parts = []
        if result["redline_count"] > 0:
            types = ", ".join(r["clause_type"].replace("_", " ") for r in result["redlines"])
            parts.append(f"{result['redline_count']} clauses flagged for renegotiation: {types}")
        if result["missing_clause_recommendations"]:
            missing_types = ", ".join(m["clause_type"].replace("_", " ") for m in result["missing_clause_recommendations"])
            parts.append(f"missing clauses to add: {missing_types}")

        return f"{document_code} negotiation summary — " + "; ".join(parts) + "."
