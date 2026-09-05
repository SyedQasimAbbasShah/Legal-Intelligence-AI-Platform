from __future__ import annotations

import re
import datetime as dt
from dataclasses import dataclass

from app.document_pipeline import DocumentTextProcessor, CLAUSE_TEMPLATES


@dataclass
class ExtractedClause:
    clause_type: str
    clause_text: str
    confidence: float
    start_offset: int
    end_offset: int


CLAUSE_KEYWORDS = {
    "payment": ["payment", "invoice", "fee", "compensation", "remuneration"],
    "termination": ["terminate", "termination", "cancel", "cancellation"],
    "confidentiality": ["confidential", "non-disclosure", "proprietary information"],
    "liability": ["liability", "liable", "indemnify", "indemnification"],
    "force_majeure": ["force majeure", "act of god", "beyond reasonable control"],
    "renewal": ["renew", "renewal", "automatically renew", "successive terms"],
    "governing_law": ["governing law", "governed by the laws", "jurisdiction"],
    "arbitration": ["arbitration", "arbitrator", "dispute resolution"],
}


class RuleBasedClauseExtractor:
    def extract(self, text: str) -> list[ExtractedClause]:
        clauses = []
        for clause_type, keywords in CLAUSE_KEYWORDS.items():
            for keyword in keywords:
                for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 150)
                    snippet = text[start:end].strip()
                    clauses.append(
                        ExtractedClause(
                            clause_type=clause_type,
                            clause_text=snippet,
                            confidence=0.75,
                            start_offset=start,
                            end_offset=end,
                        )
                    )
                    break
        return clauses


class NLPClauseClassifier:
    def __init__(self) -> None:
        self._nlp = None
        try:
            import spacy

            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                self._nlp = spacy.blank("en")
        except Exception:
            self._nlp = None

    @property
    def is_available(self) -> bool:
        return self._nlp is not None

    def _sentence_split(self, text: str) -> list[str]:
        if self._nlp is not None and "parser" in self._nlp.pipe_names:
            doc = self._nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def classify_sentence(self, sentence: str) -> tuple[str | None, float]:
        lowered = sentence.lower()
        best_type = None
        best_score = 0.0
        for clause_type, keywords in CLAUSE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in lowered)
            score = matches / max(len(keywords), 1)
            if score > best_score:
                best_score = score
                best_type = clause_type
        confidence = min(0.95, 0.5 + best_score * 0.9) if best_type else 0.0
        return best_type, confidence

    def extract(self, text: str) -> list[ExtractedClause]:
        sentences = self._sentence_split(text)
        clauses = []
        cursor = 0
        for sentence in sentences:
            clause_type, confidence = self.classify_sentence(sentence)
            start = text.find(sentence, cursor)
            if start == -1:
                start = cursor
            end = start + len(sentence)
            cursor = end
            if clause_type:
                clauses.append(
                    ExtractedClause(
                        clause_type=clause_type,
                        clause_text=sentence,
                        confidence=round(confidence, 3),
                        start_offset=start,
                        end_offset=end,
                    )
                )
        return clauses


class ClauseExtractionEngine:
    def __init__(self) -> None:
        self.rule_extractor = RuleBasedClauseExtractor()
        self.nlp_classifier = NLPClauseClassifier()
        self.processor = DocumentTextProcessor()

    def extract_clauses(self, document_text: str, method: str = "hybrid") -> list[ExtractedClause]:
        if method == "rule":
            return self.rule_extractor.extract(document_text)
        if method == "nlp":
            return self.nlp_classifier.extract(document_text)

        rule_clauses = self.rule_extractor.extract(document_text)
        nlp_clauses = self.nlp_classifier.extract(document_text)

        merged: dict[str, ExtractedClause] = {}
        for clause in rule_clauses + nlp_clauses:
            key = clause.clause_type
            if key not in merged or clause.confidence > merged[key].confidence:
                merged[key] = clause
        return sorted(merged.values(), key=lambda c: c.confidence, reverse=True)

    def find_missing_clauses(self, document_text: str) -> list[str]:
        found = {c.clause_type for c in self.extract_clauses(document_text)}
        all_types = set(CLAUSE_KEYWORDS.keys())
        return sorted(all_types - found)

    def analyze_document(self, document_code: str, document_text: str) -> dict:
        clauses = self.extract_clauses(document_text)
        missing = self.find_missing_clauses(document_text)

        return {
            "document_code": document_code,
            "analyzed_at": dt.datetime.utcnow().isoformat(),
            "clauses_found": len(clauses),
            "clause_types_present": [c.clause_type for c in clauses],
            "missing_clause_types": missing,
            "average_confidence": round(sum(c.confidence for c in clauses) / len(clauses), 3) if clauses else 0.0,
            "clauses": [
                {
                    "type": c.clause_type,
                    "text": c.clause_text,
                    "confidence": c.confidence,
                }
                for c in clauses
            ],
        }
