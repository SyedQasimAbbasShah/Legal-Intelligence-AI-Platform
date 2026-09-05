from __future__ import annotations

import re
import random
import datetime as dt
from dataclasses import dataclass

import numpy as np

from app.config import settings

DOCUMENT_TYPES = [
    "contract", "nda", "service_agreement", "employment_contract",
    "procurement_agreement", "vendor_agreement", "court_document",
    "regulatory_policy", "corporate_policy",
]

JURISDICTIONS = [
    "United States", "United Kingdom", "Germany", "France", "UAE",
    "Singapore", "India", "Canada", "Australia", "Pakistan",
]

DEPARTMENTS = [
    "Corporate Legal", "Procurement", "HR", "IP & Patents",
    "Compliance", "Vendor Management", "Litigation", "M&A",
]

CLAUSE_TEMPLATES = {
    "payment": "Payment shall be made within {days} days of invoice receipt, subject to a late fee of {pct}% per month.",
    "termination": "Either party may terminate this agreement with {days} days written notice for convenience.",
    "confidentiality": "The receiving party shall maintain confidentiality of disclosed information for a period of {years} years.",
    "liability": "Total liability under this agreement shall not exceed {pct}% of the total contract value.",
    "force_majeure": "Neither party shall be liable for delays caused by events beyond reasonable control, including natural disasters.",
    "renewal": "This agreement shall automatically renew for successive {years} year terms unless terminated with {days} days notice.",
    "governing_law": "This agreement shall be governed by the laws of {jurisdiction}, without regard to conflict of law principles.",
    "arbitration": "Any disputes arising under this agreement shall be resolved through binding arbitration in {jurisdiction}.",
}


@dataclass
class DocumentRecord:
    document_code: str
    document_type: str
    title: str
    jurisdiction: str
    counterparty: str
    department: str
    effective_date: str
    expiry_date: str
    contract_value: float
    full_text: str


class LegalDocumentSimulator:
    def __init__(self, seed: int = settings.model.random_seed) -> None:
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    def _generate_full_text(self, doc_type: str, jurisdiction: str) -> str:
        clause_order = random.sample(list(CLAUSE_TEMPLATES.keys()), k=random.randint(4, 8))
        sections = [f"AGREEMENT TYPE: {doc_type.replace('_', ' ').upper()}\n"]
        for clause_key in clause_order:
            template = CLAUSE_TEMPLATES[clause_key]
            text = template.format(
                days=random.choice([15, 30, 45, 60, 90]),
                pct=random.choice([1.5, 2.0, 5.0, 10.0, 100.0]),
                years=random.choice([1, 2, 3, 5]),
                jurisdiction=jurisdiction,
            )
            sections.append(f"[{clause_key.upper()}]\n{text}\n")
        return "\n".join(sections)

    def generate_document(self, index: int) -> DocumentRecord:
        doc_type = random.choice(DOCUMENT_TYPES)
        jurisdiction = random.choice(JURISDICTIONS)
        effective = dt.datetime.utcnow() - dt.timedelta(days=int(self._rng.integers(30, 900)))
        duration_days = int(self._rng.integers(180, 1095))
        expiry = effective + dt.timedelta(days=duration_days)

        return DocumentRecord(
            document_code=f"DOC-{index:07d}",
            document_type=doc_type,
            title=f"{doc_type.replace('_', ' ').title()} #{index}",
            jurisdiction=jurisdiction,
            counterparty=f"Vendor Corp {index % 500}",
            department=random.choice(DEPARTMENTS),
            effective_date=effective.isoformat(),
            expiry_date=expiry.isoformat(),
            contract_value=round(float(self._rng.gamma(3.0, 50000.0)), 2),
            full_text=self._generate_full_text(doc_type, jurisdiction),
        )

    def generate_batch(self, count: int) -> list[DocumentRecord]:
        return [self.generate_document(i) for i in range(count)]


class DocumentTextProcessor:
    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_sections(self, text: str) -> dict[str, str]:
        sections = {}
        matches = re.finditer(r"\[([A-Z_]+)\]\s*(.*?)(?=\[[A-Z_]+\]|\Z)", text, re.DOTALL)
        for match in matches:
            key = match.group(1).lower()
            sections[key] = self.clean_text(match.group(2))
        return sections

    def word_count(self, text: str) -> int:
        return len(text.split())

    def extract_dates(self, text: str) -> list[str]:
        pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
        return re.findall(pattern, text)


class LegalDocumentIntelligenceHub:
    def __init__(self) -> None:
        self.simulator = LegalDocumentSimulator()
        self.processor = DocumentTextProcessor()

    def ingest_batch(self, count: int = 200) -> dict:
        documents = self.simulator.generate_batch(count)
        processed = []
        for doc in documents:
            sections = self.processor.extract_sections(doc.full_text)
            processed.append(
                {
                    "document_code": doc.document_code,
                    "document_type": doc.document_type,
                    "word_count": self.processor.word_count(doc.full_text),
                    "sections_found": list(sections.keys()),
                }
            )

        type_breakdown: dict[str, int] = {}
        for doc in documents:
            type_breakdown[doc.document_type] = type_breakdown.get(doc.document_type, 0) + 1

        return {
            "ingested_at": dt.datetime.utcnow().isoformat(),
            "documents_ingested": len(documents),
            "type_breakdown": type_breakdown,
            "sample_processed": processed[:5],
            "documents": documents,
        }
