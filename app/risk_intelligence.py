from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.config import settings
from app.document_pipeline import LegalDocumentSimulator
from app.clause_extraction_engine import ClauseExtractionEngine


@dataclass
class RiskAssessment:
    document_code: str
    risk_level: str
    risk_score: float
    risk_factors: list[str]
    financial_exposure: float


HIGH_RISK_PATTERNS = ["unlimited liability", "no cap", "sole discretion", "perpetual", "irrevocable"]
AMBIGUOUS_PATTERNS = ["reasonable efforts", "as needed", "from time to time", "at its discretion", "may consider"]


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


class ClauseRiskScorer:
    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._vectorizer: TfidfVectorizer | None = None

    def _synthetic_training_set(self, n: int = 2000) -> tuple[list[str], list[int]]:
        rng = np.random.default_rng(settings.model.random_seed)
        texts, labels = [], []
        risky_phrases = HIGH_RISK_PATTERNS + AMBIGUOUS_PATTERNS
        safe_phrases = [
            "payment due within thirty days",
            "standard confidentiality obligations apply",
            "either party may terminate with notice",
            "governed by the laws of the jurisdiction",
            "liability capped at contract value",
        ]
        for _ in range(n):
            if rng.random() < 0.3:
                phrase = risky_phrases[rng.integers(0, len(risky_phrases))]
                texts.append(f"The party shall have {phrase} in fulfilling this obligation.")
                labels.append(1)
            else:
                phrase = safe_phrases[rng.integers(0, len(safe_phrases))]
                texts.append(phrase)
                labels.append(0)
        return texts, labels

    def train(self) -> dict:
        texts, labels = self._synthetic_training_set()
        self._vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X = self._vectorizer.fit_transform(texts)
        self._model = LogisticRegression(random_state=settings.model.random_seed, max_iter=500)
        self._model.fit(X, labels)
        accuracy = float(self._model.score(X, labels))
        return {"accuracy": round(accuracy, 4), "training_samples": len(texts)}

    def score_clause(self, clause_text: str) -> float:
        if self._model is None or self._vectorizer is None:
            self.train()
        vector = self._vectorizer.transform([clause_text])
        probability = float(self._model.predict_proba(vector)[0][1])
        return probability


class FinancialExposureCalculator:
    def calculate(self, contract_value: float, risk_factors: list[str]) -> float:
        multiplier = 1.0
        if "unlimited_liability" in risk_factors:
            multiplier += 2.0
        if "missing_liability_cap" in risk_factors:
            multiplier += 1.0
        if "ambiguous_language" in risk_factors:
            multiplier += 0.3
        return round(contract_value * multiplier, 2)


class ContractRiskIntelligence:
    def __init__(self) -> None:
        self.clause_scorer = ClauseRiskScorer()
        self.clause_engine = ClauseExtractionEngine()
        self.exposure_calculator = FinancialExposureCalculator()
        self._anomaly_model = IsolationForest(
            n_estimators=200, contamination=settings.model.risk_contamination, random_state=settings.model.random_seed
        )
        self._trained = False

    def initialize(self) -> dict:
        stats = self.clause_scorer.train()
        self._fit_anomaly_model()
        self._trained = True
        return stats

    def _fit_anomaly_model(self, n: int = 3000) -> None:
        rng = np.random.default_rng(settings.model.random_seed)
        features = np.column_stack(
            [
                rng.gamma(3.0, 50000.0, n),
                rng.integers(2, 10, n),
                rng.random(n),
            ]
        )
        self._anomaly_model.fit(features)

    def assess_document(self, document_code: str, document_text: str, contract_value: float) -> RiskAssessment:
        if not self._trained:
            self.initialize()

        analysis = self.clause_engine.analyze_document(document_code, document_text)
        risk_factors = []

        lowered = document_text.lower()
        for pattern in HIGH_RISK_PATTERNS:
            if pattern in lowered:
                risk_factors.append("unlimited_liability" if "liability" in pattern or "unlimited" in pattern else "high_risk_language")
                break

        for pattern in AMBIGUOUS_PATTERNS:
            if pattern in lowered:
                risk_factors.append("ambiguous_language")
                break

        if "liability" not in analysis["clause_types_present"]:
            risk_factors.append("missing_liability_cap")
        if analysis["missing_clause_types"]:
            risk_factors.append("missing_clauses")

        clause_scores = [self.clause_scorer.score_clause(c["text"]) for c in analysis["clauses"]] or [0.1]
        max_clause_risk = max(clause_scores)

        clause_count_penalty = len(analysis["missing_clause_types"]) * 0.05
        risk_score = min(1.0, max_clause_risk * 0.6 + clause_count_penalty + len(risk_factors) * 0.08)

        financial_exposure = self.exposure_calculator.calculate(contract_value, risk_factors)

        return RiskAssessment(
            document_code=document_code,
            risk_level=_risk_level(risk_score),
            risk_score=round(risk_score, 4),
            risk_factors=risk_factors or ["within_normal_parameters"],
            financial_exposure=financial_exposure,
        )

    def assess_portfolio(self, sample_size: int = 100) -> list[RiskAssessment]:
        if not self._trained:
            self.initialize()
        simulator = LegalDocumentSimulator()
        documents = simulator.generate_batch(sample_size)
        assessments = [
            self.assess_document(doc.document_code, doc.full_text, doc.contract_value) for doc in documents
        ]
        return sorted(assessments, key=lambda a: a.risk_score, reverse=True)

    def top_risk_contracts(self, limit: int = 10) -> list[RiskAssessment]:
        return self.assess_portfolio(sample_size=300)[:limit]
