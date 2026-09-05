from __future__ import annotations

import enum
import datetime as dt
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    create_engine,
    String,
    Float,
    Integer,
    DateTime,
    Enum,
    ForeignKey,
    Boolean,
    JSON,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)

from app.config import settings


class Base(DeclarativeBase):
    pass


class DocumentType(str, enum.Enum):
    CONTRACT = "contract"
    NDA = "nda"
    SERVICE_AGREEMENT = "service_agreement"
    EMPLOYMENT_CONTRACT = "employment_contract"
    PROCUREMENT_AGREEMENT = "procurement_agreement"
    VENDOR_AGREEMENT = "vendor_agreement"
    COURT_DOCUMENT = "court_document"
    REGULATORY_POLICY = "regulatory_policy"
    CORPORATE_POLICY = "corporate_policy"


class ClauseType(str, enum.Enum):
    PAYMENT = "payment"
    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    LIABILITY = "liability"
    FORCE_MAJEURE = "force_majeure"
    RENEWAL = "renewal"
    GOVERNING_LAW = "governing_law"
    ARBITRATION = "arbitration"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, enum.Enum):
    CONTRACT_EXPIRY = "contract_expiry"
    SLA_BREACH_RISK = "sla_breach_risk"
    MISSING_CLAUSE = "missing_clause"
    COMPLIANCE_ISSUE = "compliance_issue"
    VENDOR_RISK_INCREASE = "vendor_risk_increase"
    REGULATORY_CHANGE = "regulatory_change"


class LegalDocument(Base):
    __tablename__ = "legal_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_type: Mapped[DocumentType] = mapped_column(Enum(DocumentType))
    title: Mapped[str] = mapped_column(String(256))
    jurisdiction: Mapped[str] = mapped_column(String(64), index=True)
    counterparty: Mapped[str] = mapped_column(String(128), nullable=True)
    department: Mapped[str] = mapped_column(String(64), index=True)
    effective_date: Mapped[dt.date] = mapped_column(DateTime, nullable=True)
    expiry_date: Mapped[dt.date] = mapped_column(DateTime, nullable=True)
    contract_value: Mapped[float] = mapped_column(Float, default=0.0)
    full_text: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    clauses: Mapped[list["Clause"]] = relationship(back_populates="document")
    obligations: Mapped[list["Obligation"]] = relationship(back_populates="document")
    risk_assessments: Mapped[list["RiskAssessment"]] = relationship(back_populates="document")


class Clause(Base):
    __tablename__ = "clauses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_documents.id"), index=True)
    clause_type: Mapped[ClauseType] = mapped_column(Enum(ClauseType))
    clause_text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped["LegalDocument"] = relationship(back_populates="clauses")


class Obligation(Base):
    __tablename__ = "obligations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_documents.id"), index=True)
    obligation_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(String(512))
    due_date: Mapped[dt.datetime] = mapped_column(DateTime, nullable=True)
    responsible_party: Mapped[str] = mapped_column(String(128), nullable=True)
    is_fulfilled: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped["LegalDocument"] = relationship(back_populates="obligations")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_documents.id"), index=True)
    risk_level: Mapped[RiskLevel] = mapped_column(Enum(RiskLevel))
    risk_score: Mapped[float] = mapped_column(Float)
    risk_factors: Mapped[dict] = mapped_column(JSON, default=dict)
    assessed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    document: Mapped["LegalDocument"] = relationship(back_populates="risk_assessments")


class ComplianceRecord(Base):
    __tablename__ = "compliance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_documents.id"), index=True)
    framework: Mapped[str] = mapped_column(String(128))
    is_compliant: Mapped[bool] = mapped_column(Boolean)
    violations: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType))
    severity: Mapped[RiskLevel] = mapped_column(Enum(RiskLevel))
    document_code: Mapped[str] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(String(512))
    explanation: Mapped[str] = mapped_column(String(1024), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(256))
    entity: Mapped[str] = mapped_column(String(128), nullable=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


engine = create_engine(settings.database.url, pool_pre_ping=True, pool_size=20, max_overflow=40)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
