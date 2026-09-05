from __future__ import annotations

import random
import datetime as dt
from dataclasses import dataclass

import numpy as np

from app.config import settings

FRAMEWORKS = [
    "GDPR", "SOX", "HIPAA", "Procurement Standard",
    "Data Privacy Policy", "Corporate Governance Rules", "Anti-Bribery Policy",
]

FRAMEWORK_REQUIREMENTS = {
    "GDPR": ["data_processing_clause", "consent_mechanism", "data_retention_limit"],
    "SOX": ["financial_disclosure", "audit_trail", "internal_controls"],
    "HIPAA": ["data_privacy_clause", "breach_notification", "access_controls"],
    "Procurement Standard": ["competitive_bidding", "vendor_vetting", "conflict_of_interest"],
    "Data Privacy Policy": ["data_processing_clause", "data_retention_limit"],
    "Corporate Governance Rules": ["board_approval", "audit_trail"],
    "Anti-Bribery Policy": ["conflict_of_interest", "vendor_vetting"],
}


@dataclass
class ComplianceResult:
    document_code: str
    framework: str
    is_compliant: bool
    violations: list[str]
    compliance_score: float


class PolicyRuleEngine:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(settings.model.random_seed)
        random.seed(settings.model.random_seed)

    def evaluate(self, document_code: str, document_type: str, clause_types_present: list[str]) -> list[ComplianceResult]:
        results = []
        applicable_frameworks = self._applicable_frameworks(document_type)

        for framework in applicable_frameworks:
            requirements = FRAMEWORK_REQUIREMENTS.get(framework, [])
            present_synthetic = set(random.sample(requirements, k=random.randint(0, len(requirements))))
            missing = [r for r in requirements if r not in present_synthetic]

            compliance_score = 1.0 - (len(missing) / max(len(requirements), 1))
            results.append(
                ComplianceResult(
                    document_code=document_code,
                    framework=framework,
                    is_compliant=len(missing) == 0,
                    violations=missing,
                    compliance_score=round(compliance_score, 3),
                )
            )
        return results

    def _applicable_frameworks(self, document_type: str) -> list[str]:
        mapping = {
            "contract": ["SOX", "Corporate Governance Rules"],
            "nda": ["Data Privacy Policy", "GDPR"],
            "service_agreement": ["Procurement Standard", "SOX"],
            "employment_contract": ["Corporate Governance Rules", "Data Privacy Policy"],
            "procurement_agreement": ["Procurement Standard", "Anti-Bribery Policy"],
            "vendor_agreement": ["Procurement Standard", "Anti-Bribery Policy"],
            "court_document": ["SOX"],
            "regulatory_policy": ["GDPR", "HIPAA"],
            "corporate_policy": ["Corporate Governance Rules"],
        }
        return mapping.get(document_type, ["Corporate Governance Rules"])


class RegulatoryChangeMonitor:
    def check_recent_changes(self) -> list[dict]:
        changes = [
            {
                "framework": "GDPR",
                "change_summary": "Updated data retention requirements for cross-border transfers",
                "effective_date": (dt.datetime.utcnow() + dt.timedelta(days=45)).isoformat(),
                "impact_level": "high",
            },
            {
                "framework": "SOX",
                "change_summary": "New internal control disclosure requirements for Q3 filings",
                "effective_date": (dt.datetime.utcnow() + dt.timedelta(days=90)).isoformat(),
                "impact_level": "medium",
            },
            {
                "framework": "Procurement Standard",
                "change_summary": "Revised vendor vetting thresholds for contracts above $500,000",
                "effective_date": (dt.datetime.utcnow() + dt.timedelta(days=30)).isoformat(),
                "impact_level": "medium",
            },
        ]
        return changes


class ComplianceIntelligenceEngine:
    def __init__(self) -> None:
        self.rule_engine = PolicyRuleEngine()
        self.change_monitor = RegulatoryChangeMonitor()

    def evaluate_document(self, document_code: str, document_type: str, clause_types_present: list[str]) -> dict:
        results = self.rule_engine.evaluate(document_code, document_type, clause_types_present)
        overall_compliant = all(r.is_compliant for r in results)
        return {
            "document_code": document_code,
            "evaluated_at": dt.datetime.utcnow().isoformat(),
            "overall_compliant": overall_compliant,
            "frameworks_evaluated": len(results),
            "results": [
                {
                    "framework": r.framework,
                    "is_compliant": r.is_compliant,
                    "violations": r.violations,
                    "compliance_score": r.compliance_score,
                }
                for r in results
            ],
        }

    def evaluate_portfolio(self, sample_size: int = 100) -> dict:
        from app.document_pipeline import LegalDocumentSimulator
        from app.clause_extraction_engine import ClauseExtractionEngine

        simulator = LegalDocumentSimulator()
        clause_engine = ClauseExtractionEngine()
        documents = simulator.generate_batch(sample_size)

        evaluations = []
        for doc in documents:
            clauses = clause_engine.extract_clauses(doc.full_text)
            clause_types = [c.clause_type for c in clauses]
            evaluations.append(self.evaluate_document(doc.document_code, doc.document_type, clause_types))

        non_compliant = [e for e in evaluations if not e["overall_compliant"]]
        framework_violation_counts: dict[str, int] = {}
        for e in evaluations:
            for r in e["results"]:
                if not r["is_compliant"]:
                    framework_violation_counts[r["framework"]] = framework_violation_counts.get(r["framework"], 0) + 1

        return {
            "evaluated_at": dt.datetime.utcnow().isoformat(),
            "documents_evaluated": len(evaluations),
            "compliant_count": len(evaluations) - len(non_compliant),
            "non_compliant_count": len(non_compliant),
            "compliance_rate_pct": round(100 * (len(evaluations) - len(non_compliant)) / max(len(evaluations), 1), 2),
            "framework_violation_counts": framework_violation_counts,
            "non_compliant_samples": non_compliant[:15],
        }

    def get_regulatory_changes(self) -> list[dict]:
        return self.change_monitor.check_recent_changes()
