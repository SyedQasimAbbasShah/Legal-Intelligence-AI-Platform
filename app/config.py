from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("PG_HOST", "localhost")
    port: int = int(os.getenv("PG_PORT", "5432"))
    user: str = os.getenv("PG_USER", "legalintel")
    password: str = os.getenv("PG_PASSWORD", "legalintel")
    database: str = os.getenv("PG_DATABASE", "legal_intelligence_platform")
    sslmode: str = os.getenv("PG_SSLMODE", "prefer")

    @property
    def url(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}"


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "neo4j_password")


@dataclass(frozen=True)
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str = os.getenv("REDIS_PASSWORD", "")
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"


@dataclass(frozen=True)
class OrgScaleConfig:
    legal_documents: int = 8_000_000
    active_contracts: int = 2_000_000
    countries: int = 45
    legal_departments: int = 12
    external_law_firms: int = 300
    regulatory_frameworks: int = 500


@dataclass(frozen=True)
class ModelConfig:
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    embedding_dim: int = 256
    risk_contamination: float = 0.08
    random_seed: int = 42
    max_summary_tokens: int = 400


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1536"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass(frozen=True)
class ServiceConfig:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")


@dataclass(frozen=True)
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    org_scale: OrgScaleConfig = field(default_factory=OrgScaleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)


settings = Settings()
