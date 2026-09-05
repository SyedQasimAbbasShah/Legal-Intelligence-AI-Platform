from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClauseFactorContribution:
    factor_name: str
    contribution: float
    description: str


class RiskExplainer:
    def explain_risk_score(self, risk_factors: list[str], risk_score: float, financial_exposure: float) -> str:
        if not risk_factors or risk_factors == ["within_normal_parameters"]:
            return f"Risk score {risk_score:.2f} reflects standard contract terms with no significant risk indicators detected."

        factor_descriptions = {
            "unlimited_liability": "unlimited or uncapped liability exposure",
            "missing_liability_cap": "absence of a liability cap clause",
            "ambiguous_language": "ambiguous or discretionary language",
            "missing_clauses": "one or more standard clauses missing from the document",
            "high_risk_language": "high-risk contractual language patterns",
        }
        factor_text = "; ".join(factor_descriptions.get(f, f.replace("_", " ")) for f in risk_factors)

        return (
            f"Risk score {risk_score:.2f} ({self._level_label(risk_score)}) driven by: {factor_text}. "
            f"Estimated financial exposure: ${financial_exposure:,.2f}."
        )

    def _level_label(self, score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.5:
            return "high"
        if score >= 0.25:
            return "medium"
        return "low"

    def explain_missing_clauses(self, missing_types: list[str]) -> str:
        if not missing_types:
            return "All standard clause types were identified in this document."
        readable = ", ".join(m.replace("_", " ") for m in missing_types)
        return f"The following standard clause types were not detected and should be reviewed: {readable}."

    def explain_compliance_violation(self, framework: str, violations: list[str]) -> str:
        if not violations:
            return f"Document fully satisfies {framework} requirements."
        readable = ", ".join(v.replace("_", " ") for v in violations)
        return f"Document fails {framework} compliance due to missing: {readable}."

    def explain_clause_classification(self, clause_type: str, confidence: float, matched_keywords: list[str]) -> str:
        keyword_text = ", ".join(matched_keywords) if matched_keywords else "contextual language patterns"
        return (
            f"Classified as '{clause_type.replace('_', ' ')}' with {confidence:.0%} confidence, "
            f"based on presence of: {keyword_text}."
        )

    def explain_obligation_priority(self, days_remaining: int, obligation_type: str) -> str:
        urgency = "critical" if days_remaining <= 7 else "high" if days_remaining <= 30 else "moderate"
        return (
            f"{obligation_type.replace('_', ' ').title()} due in {days_remaining} days is flagged as "
            f"{urgency} priority based on standard legal deadline thresholds."
        )
