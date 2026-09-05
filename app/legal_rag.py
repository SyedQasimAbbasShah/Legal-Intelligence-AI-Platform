from __future__ import annotations

import re
import datetime as dt
from collections import Counter
from dataclasses import dataclass

import numpy as np
import faiss


@dataclass
class LegalDocument:
    doc_id: str
    category: str
    text: str
    metadata: dict


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfEmbedder:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, documents: list[str]) -> np.ndarray:
        doc_tokens = [_tokenize(d) for d in documents]
        vocab_set = sorted({token for tokens in doc_tokens for token in tokens})
        self.vocabulary = {token: i for i, token in enumerate(vocab_set)}

        doc_count = len(documents)
        doc_freq = np.zeros(len(vocab_set))
        for tokens in doc_tokens:
            for token in set(tokens):
                doc_freq[self.vocabulary[token]] += 1
        self.idf = np.log((doc_count + 1) / (doc_freq + 1)) + 1

        return self.transform(documents)

    def transform(self, documents: list[str]) -> np.ndarray:
        vectors = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float32)
        for i, doc in enumerate(documents):
            tokens = _tokenize(doc)
            counts = Counter(tokens)
            total = max(len(tokens), 1)
            for token, count in counts.items():
                if token in self.vocabulary:
                    tf = count / total
                    vectors[i, self.vocabulary[token]] = tf * self.idf[self.vocabulary[token]]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class LegalRAGStore:
    def __init__(self) -> None:
        self.embedder = TfidfEmbedder()
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[LegalDocument] = []

    def build_index(self, documents: list[LegalDocument]) -> dict:
        self.documents = documents
        texts = [d.text for d in documents]
        vectors = self.embedder.fit(texts)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        return {"documents_indexed": len(documents), "vector_dimension": dimension}

    def search(self, query: str, top_k: int = 5, category_filter: str | None = None) -> list[dict]:
        if self.index is None or not self.documents:
            return []
        query_vector = self.embedder.transform([query])
        scores, indices = self.index.search(query_vector, min(top_k * 3, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if category_filter and doc.category != category_filter:
                continue
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "category": doc.category,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "relevance_score": round(float(score), 4),
                }
            )
            if len(results) >= top_k:
                break
        return results


class ContractComparisonEngine:
    def __init__(self) -> None:
        self.embedder = TfidfEmbedder()

    def compare(self, text_a: str, text_b: str) -> dict:
        vectors = self.embedder.fit([text_a, text_b])
        similarity = float(np.dot(vectors[0], vectors[1]))

        tokens_a = set(_tokenize(text_a))
        tokens_b = set(_tokenize(text_b))
        unique_to_a = tokens_a - tokens_b
        unique_to_b = tokens_b - tokens_a
        common = tokens_a & tokens_b

        return {
            "similarity_score": round(similarity, 4),
            "common_terms_count": len(common),
            "unique_to_document_a": len(unique_to_a),
            "unique_to_document_b": len(unique_to_b),
            "verdict": "highly_similar" if similarity > 0.7 else "moderately_similar" if similarity > 0.4 else "substantially_different",
        }


def build_default_legal_knowledge_base() -> list[LegalDocument]:
    now = dt.datetime.utcnow()
    return [
        LegalDocument(
            "precedent-001", "risk",
            "A vendor agreement was flagged for unlimited liability exposure after review found no cap on "
            "damages clause, requiring immediate renegotiation with the counterparty before execution.",
            {"date": (now - dt.timedelta(days=12)).isoformat()},
        ),
        LegalDocument(
            "precedent-002", "compliance",
            "GDPR compliance review identified missing data retention limits in three vendor agreements "
            "processed in the European jurisdiction, triggering mandatory remediation within 30 days.",
            {"framework": "GDPR", "date": (now - dt.timedelta(days=8)).isoformat()},
        ),
        LegalDocument(
            "policy-001", "policy",
            "Company procurement policy requires competitive bidding for all vendor agreements exceeding "
            "500,000 dollars in total contract value, with documented vendor vetting records retained.",
            {"category": "procurement"},
        ),
        LegalDocument(
            "policy-002", "policy",
            "Standard confidentiality clause requires a minimum five year non-disclosure period for all "
            "NDAs involving proprietary technical information or trade secrets.",
            {"category": "confidentiality"},
        ),
        LegalDocument(
            "precedent-003", "obligation",
            "Contract renewal notice periods of 60 days are standard for service agreements, with automatic "
            "renewal triggered unless written notice is delivered before the deadline.",
            {"date": (now - dt.timedelta(days=20)).isoformat()},
        ),
        LegalDocument(
            "guide-001", "clause_guidance",
            "Force majeure clauses should explicitly enumerate qualifying events including natural disasters, "
            "government action, and pandemics to avoid ambiguous interpretation during disputes.",
            {"category": "drafting_guide"},
        ),
        LegalDocument(
            "precedent-004", "litigation",
            "Arbitration clauses specifying jurisdiction outside the company headquarters increased litigation "
            "costs by an estimated 40 percent in prior disputes, prompting a policy preference for domestic arbitration.",
            {"date": (now - dt.timedelta(days=30)).isoformat()},
        ),
        LegalDocument(
            "guide-002", "clause_guidance",
            "Termination for convenience clauses should specify a notice period of at least 30 days and "
            "clarify any wind-down obligations for outstanding deliverables.",
            {"category": "drafting_guide"},
        ),
    ]
