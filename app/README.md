# Enterprise Legal Intelligence & Contract Reasoning Platform
### Ezitech Engineering Framework — Case Study AI-234

AI-driven platform for analyzing legal documents, extracting clauses, identifying contractual
risks, tracking obligations, evaluating compliance, and assisting legal teams through a
conversational copilot.

## Architecture

```
                    ┌────────────────────────────┐
                    │   Legal Document Sources     │
                    │  Contracts / NDAs / Policies │
                    │  Vendor & Employment Agreemts │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │ Legal Document Intelligence │
                    │  (document_pipeline.py)     │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │   Event Bus (Redis Pub/Sub)  │
                    │   (event_bus.py)             │
                    │ DOCUMENT_INGESTED → reactive  │
                    │ chain: clause extraction →    │
                    │ risk scoring → alert raised   │
                    └──────────────┬─────────────┘
                                   │
        ┌──────────────┬──────────┴──────────┬──────────────┐
        ▼              ▼                     ▼              ▼
 ┌─────────────┐ ┌───────────┐      ┌────────────────┐ ┌───────────┐
 │   Clause    │ │   Risk    │      │   Obligation    │ │Compliance │
 │ Extraction  │ │Intelligence│      │    Tracking     │ │  Engine   │
 │ spaCy + NLP │ │TF-IDF+LogReg│    │  Deadline Monitor│ │Rule Engine│
 └──────┬──────┘ └─────┬─────┘      └────────┬────────┘ └─────┬─────┘
        └───────────────┴──────────────────────┴────────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │   PostgreSQL + Redis         │
                    │   (database.py)              │
                    └──────────────┬─────────────┘
                                   │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 ┌──────────────┐         ┌──────────────────┐        ┌──────────────────┐
 │Knowledge Graph│         │  Explainable AI    │        │   Alert Center    │
 │Neo4j/NetworkX │         │  Risk narratives   │        │ (alert_center.py) │
 └──────┬───────┘         └─────────┬─────────┘        └─────────┬─────────┘
        └───────────────────────────┼───────────────────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │   AI Legal Copilot (Multi-Agent)│
                    │   Supervisor routes to:      │
                    │   Contract Analyst · Risk     │
                    │   Analyst · Compliance Officer│
                    │   · Obligations Tracker ·     │
                    │   Research Analyst (RAG/FAISS)│
                    │   (legal_copilot.py)         │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │   FastAPI Gateway             │
                    │   (api_gateway.py)            │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │   Compliance Dashboard        │
                    │   (dashboard.html)            │
                    └───────────────────────────┘
```

## Module Map

| File | Core Module (per spec) | Techniques Used |
|---|---|---|
| `config.py` | Platform configuration | Env-driven dataclasses, SSL-ready DB/Redis |
| `database.py` | Data model / persistence | SQLAlchemy ORM, PostgreSQL |
| `document_pipeline.py` | Legal Document Intelligence | Synthetic corpus generation, text processing |
| `event_bus.py` | Event-Driven Architecture | Redis Pub/Sub with in-process fallback, reactive processing chain |
| `clause_extraction_engine.py` | Clause Extraction Engine | Rule-based patterns + spaCy NLP hybrid |
| `risk_intelligence.py` | Contract Risk Intelligence | TF-IDF + Logistic Regression, Isolation Forest |
| `obligation_tracking.py` | Obligation Tracking Engine | Deadline extraction and monitoring |
| `knowledge_graph.py` | Legal Knowledge Graph | NetworkX, Neo4j driver, vendor exposure ranking |
| `compliance_engine.py` | Compliance Intelligence | Policy rule engine, regulatory change monitor |
| `legal_rag.py` | Legal RAG | FAISS TF-IDF vector search, contract comparison |
| `explainable_ai.py` | Explainable AI | Human-readable risk/compliance narratives |
| `legal_copilot.py` | AI Legal Copilot / Multi-Agent AI Workflow | LangGraph supervisor routing to 5 specialist agents (contract analyst, risk analyst, compliance officer, obligations tracker, research analyst), fan-out/converge orchestration |
| `alert_center.py` | Alert Center | Cross-engine rule aggregation |
| `auth.py` | Enterprise Security | API-key auth, role scopes, rate limiting |
| `audit.py` | Complete Audit Logging | Middleware logging every request |
| `cache.py` | Performance / caching | Redis with in-memory fallback |
| `bonus_contract_negotiation.py` | Bonus: AI Negotiation Assistant | Clause playbook, redline suggestions |
| `api_gateway.py` | REST APIs / API Gateway | FastAPI, CORS, auth, rate limiting, caching |
| `dashboard.html` | Compliance Dashboard | Chart.js, live copilot integration |
| `tests/test_engines.py` | Testing / QA | pytest unit tests across all engines |
| `demo_script.py` | Live Demonstration deliverable | End-to-end REST API walkthrough |
| `Dockerfile` / `docker-compose.yml` | Deployment | Full containerized stack |

## AI Stack Coverage

| PDF AI Stack Item | Used In |
|---|---|
| LangGraph | `legal_copilot.py` — StateGraph supervisor pattern: fan-out to specialist agents, converge at synthesizer |
| LangChain | Not used directly — LangGraph alone covers agent orchestration without the extra dependency weight |
| LlamaIndex | Not used — FAISS-based `legal_rag.py` covers retrieval without an additional framework |
| Hugging Face Transformers | Not used — TF-IDF embeddings keep the RAG pipeline dependency-free and fully offline |
| Sentence Transformers | Same as above |
| FAISS / ChromaDB | `legal_rag.py` — FAISS `IndexFlatIP` over TF-IDF vectors |
| OpenAI or Open Source LLM | `legal_copilot.py` — Anthropic Claude, with deterministic fallback if no API key |
| spaCy | `clause_extraction_engine.py` — sentence segmentation and clause classification |
| NetworkX | `knowledge_graph.py` |

**Note:** LangChain, LlamaIndex, Hugging Face Transformers, and Sentence Transformers were
intentionally left out in favor of lighter, dependency-free equivalents (LangGraph alone,
FAISS+TF-IDF) that deliver the same functional capability — multi-agent orchestration and
semantic retrieval — without requiring model downloads or a heavier dependency tree, keeping
the platform easy to run in constrained or offline deployment environments.

## Requirements

- Python **3.13**
- PostgreSQL 15+
- Neo4j 5+ (optional — falls back to NetworkX-only if unavailable)
- Redis 7+ (optional caching layer, falls back to in-memory)
- An `ANTHROPIC_API_KEY` environment variable for the AI Legal Copilot LLM calls (falls back
  to a deterministic rule-based responder if not set)

## Deployment Guide

1. **Create environment**
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your database, Redis, and API key values. Set a fixed `LEGALINTEL_API_KEYS`
   value so it does not regenerate randomly on every server restart.

3. **Provision PostgreSQL**
   ```bash
   createdb legal_intelligence_platform
   ```
   Tables are created automatically on API startup via `init_db()`.

4. **Run the API Gateway**
   ```bash
   uvicorn app.api_gateway:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open the Compliance Dashboard**
   Visit `http://localhost:8000/dashboard` — served directly by the backend, so the AI Copilot
   box connects automatically without any manual URL configuration.

6. **Seed the knowledge graph** (optional)
   ```bash
   curl -X POST "http://localhost:8000/knowledge-graph/seed?num_contracts=50" -H "X-API-Key: your-key"
   ```

7. **Run the test suite**
   ```bash
   pytest tests/ -v
   ```

8. **Run the full end-to-end demo**
   ```bash
   export LEGALINTEL_API_URL=http://localhost:8000
   export LEGALINTEL_DEMO_API_KEY=your-key
   python demo_script.py
   ```

9. **Docker Compose (full stack in one command)**
   ```bash
   docker compose up --build
   ```

### Authentication

All endpoints except `/health` and `/dashboard` require an `X-API-Key` header. Configure keys
via `LEGALINTEL_API_KEYS` as `client_id:role:raw_key` pairs, comma-separated. Roles: `paralegal`
(read), `attorney` (read+write), `general_counsel` (read+write+admin).

## Scaling Notes

- The platform is stateless and can run behind a load balancer with multiple Uvicorn workers.
- All engines degrade gracefully: if Neo4j, Redis, or the LLM provider are unreachable, each
  module falls back to a local/heuristic mode so the platform remains demonstrable end-to-end.
- Designed for horizontal scaling to support the 8M document / 2M active contract scale
  described in the case study; document ingestion and clause extraction are stateless and can
  be parallelized across workers.

## Bonus Challenge

**AI Contract Negotiation Assistant** is fully implemented in `bonus_contract_negotiation.py`:
a clause-level playbook detects risky language (unlimited liability, ambiguous discretion,
perpetual terms) and generates redline suggestions with rationale and negotiation leverage
points for each flagged clause, plus recommendations for any missing standard clauses.
Exposed via `POST /bonus/negotiation-redlines`.

Other bonus ideas remain natural extensions of the existing pipeline:
- **Regulatory Change Impact Analyzer**: extend `compliance_engine.py`'s `RegulatoryChangeMonitor`
  to cross-reference changes against the contract portfolio for affected-document detection.
- **AI Litigation Risk Predictor**: extend `risk_intelligence.py` with a classifier trained on
  historical dispute outcomes tied to specific clause language patterns.
