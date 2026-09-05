from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.database import Alert, AlertType, RiskLevel, get_session, AuditLog
from app.obligation_tracking import ObligationTrackingEngine
from app.risk_intelligence import ContractRiskIntelligence
from app.compliance_engine import ComplianceIntelligenceEngine
from app.knowledge_graph import LegalKnowledgeGraph
from app.explainable_ai import RiskExplainer


@dataclass
class AlertRecord:
    alert_type: AlertType
    severity: RiskLevel
    document_code: str | None
    message: str
    explanation: str
    created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)


class LegalAlertCenter:
    def __init__(self) -> None:
        self.obligation_engine = ObligationTrackingEngine()
        self.risk_engine = ContractRiskIntelligence()
        self.compliance_engine = ComplianceIntelligenceEngine()
        self.knowledge_graph = LegalKnowledgeGraph()
        self.explainer = RiskExplainer()

    def generate_expiry_alerts(self) -> list[AlertRecord]:
        tracking = self.obligation_engine.track_portfolio(sample_size=150)
        alerts = []
        for contract in tracking["expiring_contracts"]:
            days = contract["days_remaining"]
            severity = RiskLevel.CRITICAL if days <= 14 else RiskLevel.HIGH if days <= 30 else RiskLevel.MEDIUM
            alerts.append(
                AlertRecord(
                    alert_type=AlertType.CONTRACT_EXPIRY,
                    severity=severity,
                    document_code=contract["document_code"],
                    message=f"{contract['document_code']} expires in {days} days",
                    explanation=self.explainer.explain_obligation_priority(days, "contract_expiry"),
                )
            )
        return alerts

    def generate_sla_alerts(self) -> list[AlertRecord]:
        tracking = self.obligation_engine.track_portfolio(sample_size=150)
        alerts = []
        for obligation in tracking["overdue_obligations"]:
            if obligation["type"] != "sla_milestone":
                continue
            alerts.append(
                AlertRecord(
                    alert_type=AlertType.SLA_BREACH_RISK,
                    severity=RiskLevel.HIGH,
                    document_code=obligation["document_code"],
                    message=f"SLA milestone overdue on {obligation['document_code']}",
                    explanation=f"Obligation was due {obligation['due_date']} and remains unfulfilled.",
                )
            )
        return alerts

    def generate_missing_clause_alerts(self, limit: int = 10) -> list[AlertRecord]:
        assessments = self.risk_engine.top_risk_contracts(limit=limit)
        alerts = []
        for a in assessments:
            if "missing_clauses" not in a.risk_factors:
                continue
            alerts.append(
                AlertRecord(
                    alert_type=AlertType.MISSING_CLAUSE,
                    severity=RiskLevel.MEDIUM,
                    document_code=a.document_code,
                    message=f"{a.document_code} is missing one or more standard clauses",
                    explanation=self.explainer.explain_risk_score(a.risk_factors, a.risk_score, a.financial_exposure),
                )
            )
        return alerts

    def generate_compliance_alerts(self) -> list[AlertRecord]:
        portfolio = self.compliance_engine.evaluate_portfolio(sample_size=80)
        alerts = []
        for sample in portfolio["non_compliant_samples"][:10]:
            failed_frameworks = [r["framework"] for r in sample["results"] if not r["is_compliant"]]
            alerts.append(
                AlertRecord(
                    alert_type=AlertType.COMPLIANCE_ISSUE,
                    severity=RiskLevel.HIGH,
                    document_code=sample["document_code"],
                    message=f"{sample['document_code']} fails compliance for {', '.join(failed_frameworks)}",
                    explanation=self.explainer.explain_compliance_violation(
                        failed_frameworks[0] if failed_frameworks else "policy",
                        sample["results"][0]["violations"] if sample["results"] else [],
                    ),
                )
            )
        return alerts

    def generate_vendor_risk_alerts(self) -> list[AlertRecord]:
        self.knowledge_graph.seed_sample_graph(num_contracts=40)
        high_exposure = self.knowledge_graph.find_high_exposure_vendors(top_k=5)
        alerts = []
        for v in high_exposure:
            if v["total_exposure"] < 200000:
                continue
            alerts.append(
                AlertRecord(
                    alert_type=AlertType.VENDOR_RISK_INCREASE,
                    severity=RiskLevel.MEDIUM,
                    document_code=None,
                    message=f"{v['vendor']} has total contract exposure of ${v['total_exposure']:,.2f}",
                    explanation="Vendor exposure exceeds internal concentration risk threshold and should be reviewed.",
                )
            )
        return alerts

    def generate_regulatory_alerts(self) -> list[AlertRecord]:
        changes = self.compliance_engine.get_regulatory_changes()
        alerts = []
        for change in changes:
            severity = RiskLevel.HIGH if change["impact_level"] == "high" else RiskLevel.MEDIUM
            alerts.append(
                AlertRecord(
                    alert_type=AlertType.REGULATORY_CHANGE,
                    severity=severity,
                    document_code=None,
                    message=f"{change['framework']}: {change['change_summary']}",
                    explanation=f"Effective {change['effective_date']}, impact level: {change['impact_level']}.",
                )
            )
        return alerts

    def run_full_sweep(self) -> dict:
        all_alerts = (
            self.generate_expiry_alerts()
            + self.generate_sla_alerts()
            + self.generate_missing_clause_alerts()
            + self.generate_compliance_alerts()
            + self.generate_vendor_risk_alerts()
            + self.generate_regulatory_alerts()
        )
        self._persist(all_alerts)

        severity_counts: dict[str, int] = {}
        for a in all_alerts:
            severity_counts[a.severity.value] = severity_counts.get(a.severity.value, 0) + 1

        return {
            "swept_at": dt.datetime.utcnow().isoformat(),
            "total_alerts": len(all_alerts),
            "severity_breakdown": severity_counts,
            "alerts": [
                {
                    "type": a.alert_type.value,
                    "severity": a.severity.value,
                    "document_code": a.document_code,
                    "message": a.message,
                    "explanation": a.explanation,
                    "created_at": a.created_at.isoformat(),
                }
                for a in all_alerts
            ],
        }

    def _persist(self, alerts: list[AlertRecord]) -> None:
        try:
            with get_session() as session:
                for a in alerts:
                    session.add(
                        Alert(
                            alert_type=a.alert_type,
                            severity=a.severity,
                            document_code=a.document_code,
                            message=a.message,
                            explanation=a.explanation,
                            created_at=a.created_at,
                        )
                    )
                session.add(
                    AuditLog(
                        actor="alert_center",
                        action="generate_alerts",
                        entity="alerts",
                        details={"count": len(alerts)},
                    )
                )
        except Exception:
            pass
