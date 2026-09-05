from __future__ import annotations

import re
import random
import datetime as dt
from dataclasses import dataclass

import numpy as np

from app.config import settings


@dataclass
class ObligationRecord:
    document_code: str
    obligation_type: str
    description: str
    due_date: dt.datetime
    responsible_party: str
    is_fulfilled: bool


OBLIGATION_TYPES = [
    "renewal_date", "deliverable", "payment_deadline",
    "notice_period", "compliance_requirement", "sla_milestone",
]

OBLIGATION_TEMPLATES = {
    "renewal_date": "Contract renewal decision due before {days} days prior to expiry",
    "deliverable": "Deliverable milestone {n} due for review and acceptance",
    "payment_deadline": "Invoice payment of ${amount} due to counterparty",
    "notice_period": "Termination notice window of {days} days must be honored",
    "compliance_requirement": "Annual compliance certification must be filed",
    "sla_milestone": "Service level target of {pct}% uptime must be reported",
}


class ObligationExtractor:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(settings.model.random_seed)
        random.seed(settings.model.random_seed)

    def extract_from_document(
        self, document_code: str, effective_date: dt.datetime, expiry_date: dt.datetime, contract_value: float
    ) -> list[ObligationRecord]:
        obligations = []
        num_obligations = random.randint(2, 6)
        span_days = max((expiry_date - effective_date).days, 1)

        for _ in range(num_obligations):
            obligation_type = random.choice(OBLIGATION_TYPES)
            template = OBLIGATION_TEMPLATES[obligation_type]
            description = template.format(
                days=random.choice([30, 60, 90]),
                n=random.randint(1, 5),
                amount=round(float(self._rng.uniform(1000, contract_value * 0.2 + 1000)), 2),
                pct=random.choice([95, 99, 99.9]),
            )
            offset_days = int(self._rng.integers(0, span_days))
            due_date = effective_date + dt.timedelta(days=offset_days)

            obligations.append(
                ObligationRecord(
                    document_code=document_code,
                    obligation_type=obligation_type,
                    description=description,
                    due_date=due_date,
                    responsible_party=random.choice(["Internal Legal", "Vendor", "Finance", "Procurement"]),
                    is_fulfilled=due_date < dt.datetime.utcnow() and random.random() > 0.15,
                )
            )
        return obligations


class DeadlineMonitor:
    def find_upcoming(self, obligations: list[ObligationRecord], within_days: int = 30) -> list[ObligationRecord]:
        now = dt.datetime.utcnow()
        horizon = now + dt.timedelta(days=within_days)
        return [
            o for o in obligations
            if not o.is_fulfilled and now <= o.due_date <= horizon
        ]

    def find_overdue(self, obligations: list[ObligationRecord]) -> list[ObligationRecord]:
        now = dt.datetime.utcnow()
        return [o for o in obligations if not o.is_fulfilled and o.due_date < now]

    def find_expiring_contracts(
        self, contract_expiries: list[tuple[str, dt.datetime]], within_days: int = 60
    ) -> list[dict]:
        now = dt.datetime.utcnow()
        horizon = now + dt.timedelta(days=within_days)
        expiring = []
        for code, expiry in contract_expiries:
            if now <= expiry <= horizon:
                expiring.append(
                    {
                        "document_code": code,
                        "expiry_date": expiry.isoformat(),
                        "days_remaining": (expiry - now).days,
                    }
                )
        return sorted(expiring, key=lambda e: e["days_remaining"])


class ObligationTrackingEngine:
    def __init__(self) -> None:
        self.extractor = ObligationExtractor()
        self.monitor = DeadlineMonitor()

    def track_portfolio(self, sample_size: int = 150) -> dict:
        from app.document_pipeline import LegalDocumentSimulator

        simulator = LegalDocumentSimulator()
        documents = simulator.generate_batch(sample_size)

        all_obligations = []
        contract_expiries = []
        for doc in documents:
            effective = dt.datetime.fromisoformat(doc.effective_date)
            expiry = dt.datetime.fromisoformat(doc.expiry_date)
            obligations = self.extractor.extract_from_document(doc.document_code, effective, expiry, doc.contract_value)
            all_obligations.extend(obligations)
            contract_expiries.append((doc.document_code, expiry))

        upcoming = self.monitor.find_upcoming(all_obligations)
        overdue = self.monitor.find_overdue(all_obligations)
        expiring_contracts = self.monitor.find_expiring_contracts(contract_expiries)

        return {
            "tracked_at": dt.datetime.utcnow().isoformat(),
            "total_obligations": len(all_obligations),
            "upcoming_30_days": len(upcoming),
            "overdue_count": len(overdue),
            "expiring_contracts_60_days": len(expiring_contracts),
            "upcoming_obligations": [
                {
                    "document_code": o.document_code,
                    "type": o.obligation_type,
                    "description": o.description,
                    "due_date": o.due_date.isoformat(),
                    "responsible_party": o.responsible_party,
                }
                for o in upcoming[:20]
            ],
            "overdue_obligations": [
                {
                    "document_code": o.document_code,
                    "type": o.obligation_type,
                    "description": o.description,
                    "due_date": o.due_date.isoformat(),
                }
                for o in overdue[:20]
            ],
            "expiring_contracts": expiring_contracts[:20],
        }
