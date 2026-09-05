from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass
from typing import Callable, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from app.config import settings
from app.document_pipeline import LegalDocumentSimulator
from app.clause_extraction_engine import ClauseExtractionEngine
from app.risk_intelligence import ContractRiskIntelligence
from app.obligation_tracking import ObligationTrackingEngine
from app.compliance_engine import ComplianceIntelligenceEngine
from app.legal_rag import LegalRAGStore, ContractComparisonEngine, build_default_legal_knowledge_base


@dataclass
class CopilotResponse:
    answer: str
    tool_calls: list[str]
    agents_consulted: list[str]
    generated_at: dt.datetime


AGENT_TOOLS = {
    "contract_analyst": {
        "name": "summarize_contract",
        "description": "Summarize a contract's key terms, clauses, and obligations",
        "input_schema": {
            "type": "object",
            "properties": {"document_code": {"type": "string"}},
            "required": ["document_code"],
        },
    },
    "risk_analyst": {
        "name": "highlight_risky_clauses",
        "description": "Identify the highest risk contracts and their risk factors",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    "compliance_officer": {
        "name": "check_policy_violations",
        "description": "Check which contracts violate company compliance policy",
        "input_schema": {"type": "object", "properties": {}},
    },
    "obligations_tracker": {
        "name": "list_expiring_obligations",
        "description": "List obligations and contracts expiring within a time window",
        "input_schema": {
            "type": "object",
            "properties": {"within_days": {"type": "integer"}},
        },
    },
    "research_analyst": {
        "name": "search_legal_knowledge_base",
        "description": "Semantic search over past precedents, policies, and drafting guidance",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    "clause_explainer": {
        "name": "explain_clause",
        "description": "Explain a legal clause in simple, plain language for a non-lawyer",
        "input_schema": {
            "type": "object",
            "properties": {"clause_text": {"type": "string"}},
            "required": ["clause_text"],
        },
    },
}

AGENT_SYSTEM_PROMPTS = {
    "contract_analyst": (
        "You are the Contract Analyst agent on a legal team. Summarize contract terms, "
        "clauses, and key facts concisely and accurately using the summarize_contract tool."
    ),
    "risk_analyst": (
        "You are the Risk Analyst agent on a legal team. Identify and explain high-risk "
        "contract clauses and financial exposure using the highlight_risky_clauses tool."
    ),
    "compliance_officer": (
        "You are the Compliance Officer agent on a legal team. Evaluate contracts against "
        "regulatory frameworks and internal policy using the check_policy_violations tool."
    ),
    "obligations_tracker": (
        "You are the Obligations Tracker agent on a legal team. Report on upcoming and "
        "overdue contractual obligations using the list_expiring_obligations tool."
    ),
    "research_analyst": (
        "You are the Legal Research agent on a legal team. Retrieve relevant precedents, "
        "policies, and drafting guidance using the search_legal_knowledge_base tool."
    ),
    "clause_explainer": (
        "You are the Clause Explainer agent on a legal team. Translate dense legal language "
        "into simple, plain terms a non-lawyer can understand, using the explain_clause tool."
    ),
}

AGENT_KEYWORDS = {
    "contract_analyst": ["summar", "compare", "contract terms", "what does this contract"],
    "risk_analyst": ["risk", "risky", "exposure", "liability"],
    "compliance_officer": ["complian", "polic", "violat", "regulation", "framework"],
    "obligations_tracker": ["expir", "obligation", "deadline", "due", "renewal", "notice period"],
    "research_analyst": ["precedent", "guidance", "history", "past incident"],
    "clause_explainer": ["explain", "simple language", "plain language", "plain terms", "what does this mean", "in layman"],
}


PLAIN_LANGUAGE_EXPLANATIONS = {
    "payment": "This clause says when and how you need to pay the other party, and what happens if payment is late (such as extra fees).",
    "termination": "This clause explains how either side can end the agreement early, and how much advance notice they must give.",
    "confidentiality": "This clause means both sides promise to keep shared information secret, and specifies for how long that promise lasts.",
    "liability": "This clause limits how much money one party has to pay the other if something goes wrong under the contract.",
    "force_majeure": "This clause excuses both parties from blame if unexpected events, like natural disasters, prevent them from fulfilling the contract.",
    "renewal": "This clause explains whether the contract automatically continues after it ends, and how to opt out if you don't want it to.",
    "governing_law": "This clause decides which country's or state's laws will be used to interpret and enforce the contract.",
    "arbitration": "This clause means any disputes will be settled by a private arbitrator instead of going to court.",
}


class LegalToolExecutor:
    def __init__(self) -> None:
        self.simulator = LegalDocumentSimulator()
        self.clause_engine = ClauseExtractionEngine()
        self.risk_engine = ContractRiskIntelligence()
        self.obligation_engine = ObligationTrackingEngine()
        self.compliance_engine = ComplianceIntelligenceEngine()
        self.comparison_engine = ContractComparisonEngine()
        self.vector_store = LegalRAGStore()
        self.vector_store.build_index(build_default_legal_knowledge_base())

        self._registry: dict[str, Callable] = {
            "summarize_contract": self._summarize_contract,
            "compare_agreements": self._compare_agreements,
            "highlight_risky_clauses": self._highlight_risky_clauses,
            "list_expiring_obligations": self._list_expiring_obligations,
            "check_policy_violations": self._check_policy_violations,
            "search_legal_knowledge_base": self._search_legal_knowledge_base,
            "explain_clause": self._explain_clause,
        }

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        handler = self._registry.get(tool_name)
        if not handler:
            return {"error": f"unknown tool {tool_name}"}
        return handler(tool_input)

    def _summarize_contract(self, args: dict) -> dict:
        document_code = args.get("document_code", "DOC-0000001")
        index = abs(hash(document_code)) % 1000
        document = self.simulator.generate_document(index)
        analysis = self.clause_engine.analyze_document(document.document_code, document.full_text)
        return {
            "document_code": document.document_code,
            "document_type": document.document_type,
            "counterparty": document.counterparty,
            "contract_value": document.contract_value,
            "expiry_date": document.expiry_date,
            "clauses_found": analysis["clause_types_present"],
            "missing_clauses": analysis["missing_clause_types"],
        }

    def _compare_agreements(self, args: dict) -> dict:
        return self.comparison_engine.compare(args.get("text_a", ""), args.get("text_b", ""))

    def _highlight_risky_clauses(self, args: dict) -> dict:
        limit = args.get("limit", 5)
        assessments = self.risk_engine.top_risk_contracts(limit=limit)
        return {
            "contracts": [
                {
                    "document_code": a.document_code,
                    "risk_level": a.risk_level,
                    "risk_score": a.risk_score,
                    "risk_factors": a.risk_factors,
                    "financial_exposure": a.financial_exposure,
                }
                for a in assessments
            ]
        }

    def _list_expiring_obligations(self, args: dict) -> dict:
        within_days = args.get("within_days", 30)
        tracking = self.obligation_engine.track_portfolio(sample_size=150)
        return {
            "within_days": within_days,
            "upcoming_obligations": tracking["upcoming_obligations"],
            "expiring_contracts": tracking["expiring_contracts"],
        }

    def _check_policy_violations(self, args: dict) -> dict:
        result = self.compliance_engine.evaluate_portfolio(sample_size=80)
        return {
            "non_compliant_count": result["non_compliant_count"],
            "compliance_rate_pct": result["compliance_rate_pct"],
            "framework_violation_counts": result["framework_violation_counts"],
            "samples": result["non_compliant_samples"][:5],
        }

    def _search_legal_knowledge_base(self, args: dict) -> dict:
        query = args.get("query", "")
        results = self.vector_store.search(query, top_k=3)
        return {"query": query, "results": results}

    def _explain_clause(self, args: dict) -> dict:
        clause_text = args.get("clause_text", "")
        clause_type, confidence = self.clause_engine.nlp_classifier.classify_sentence(clause_text)
        explanation = PLAIN_LANGUAGE_EXPLANATIONS.get(
            clause_type, "This clause does not match a standard category; it should be reviewed manually by an attorney."
        )
        return {
            "original_text": clause_text,
            "identified_clause_type": clause_type or "unclassified",
            "classification_confidence": round(confidence, 3),
            "plain_language_explanation": explanation,
        }


class CopilotState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]
    agents_consulted: Annotated[list, operator.add]
    tool_calls_made: Annotated[list, operator.add]
    agent_outputs: Annotated[list, operator.add]
    final_answer: str


class AILegalCopilot:
    def __init__(self) -> None:
        self.executor = LegalToolExecutor()
        self._client = None
        try:
            import anthropic

            self._client = anthropic.Anthropic()
        except Exception:
            self._client = None
        self._graph = self._build_graph()

    def _route_agents(self, question: str) -> list[str]:
        lowered = question.lower()
        matched = [
            agent for agent, keywords in AGENT_KEYWORDS.items()
            if any(kw in lowered for kw in keywords)
        ]
        return matched or ["research_analyst"]

    def _supervisor_node(self, state: CopilotState) -> dict:
        return {"messages": []}

    def _route_from_supervisor(self, state: CopilotState) -> list[str]:
        return self._route_agents(state["question"])

    def _run_specialist_agent(self, agent_name: str, question: str) -> tuple[str, list[str]]:
        tool_def = AGENT_TOOLS[agent_name]
        tool_calls_made = []

        if self._client is None:
            tool_input = self._default_tool_input(agent_name)
            result = self.executor.execute(tool_def["name"], tool_input)
            tool_calls_made.append(tool_def["name"])
            return self._summarize_result_locally(agent_name, result), tool_calls_made

        messages = [{"role": "user", "content": question}]
        response = self._client.messages.create(
            model=settings.llm.model,
            max_tokens=600,
            system=AGENT_SYSTEM_PROMPTS[agent_name],
            tools=[tool_def],
            messages=messages,
        )

        while response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for block in tool_use_blocks:
                tool_calls_made.append(block.name)
                result = self.executor.execute(block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            response = self._client.messages.create(
                model=settings.llm.model,
                max_tokens=600,
                system=AGENT_SYSTEM_PROMPTS[agent_name],
                tools=[tool_def],
                messages=messages,
            )

        text_blocks = [b.text for b in response.content if b.type == "text"]
        answer = "\n".join(text_blocks) if text_blocks else "No response generated."
        return answer, tool_calls_made

    def _default_tool_input(self, agent_name: str) -> dict:
        defaults = {
            "contract_analyst": {"document_code": "DOC-0000042"},
            "risk_analyst": {"limit": 5},
            "compliance_officer": {},
            "obligations_tracker": {"within_days": 30},
            "research_analyst": {"query": "contract risk"},
            "clause_explainer": {"clause_text": "The receiving party shall maintain confidentiality of disclosed information for a period of 5 years."},
        }
        return defaults.get(agent_name, {})

    def _summarize_result_locally(self, agent_name: str, result: dict) -> str:
        if agent_name == "contract_analyst":
            return (
                f"{result['document_code']} ({result['document_type']}) with {result['counterparty']}, "
                f"valued at ${result['contract_value']:,.2f}. Clauses present: "
                f"{', '.join(result['clauses_found']) or 'none detected'}."
            )
        if agent_name == "risk_analyst":
            top = result["contracts"][0] if result["contracts"] else None
            return (
                f"Highest risk contract: {top['document_code']} at {top['risk_score']:.0%} risk, "
                f"driven by {', '.join(top['risk_factors'])}."
                if top
                else "No high-risk contracts identified."
            )
        if agent_name == "compliance_officer":
            return f"{result['non_compliant_count']} contracts non-compliant. Compliance rate: {result['compliance_rate_pct']}%."
        if agent_name == "obligations_tracker":
            return f"{len(result['expiring_contracts'])} contracts expiring, {len(result['upcoming_obligations'])} obligations upcoming."
        if agent_name == "research_analyst":
            if result["results"]:
                top = result["results"][0]
                return f"Relevant precedent ({top['category']}): {top['text']}"
            return "No relevant knowledge base entries found."
        if agent_name == "clause_explainer":
            return f"In plain language: {result['plain_language_explanation']}"
        return "No output generated."

    def _make_agent_node(self, agent_name: str):
        def node(state: CopilotState) -> dict:
            answer, tool_calls = self._run_specialist_agent(agent_name, state["question"])
            return {
                "agents_consulted": [agent_name],
                "tool_calls_made": tool_calls,
                "agent_outputs": [{"agent": agent_name, "output": answer}],
            }

        return node

    def _synthesizer_node(self, state: CopilotState) -> dict:
        outputs = state["agent_outputs"]
        if len(outputs) == 1:
            return {"final_answer": outputs[0]["output"]}

        labels = {
            "contract_analyst": "Contract Analyst",
            "risk_analyst": "Risk Analyst",
            "compliance_officer": "Compliance Officer",
            "obligations_tracker": "Obligations Tracker",
            "research_analyst": "Legal Research",
        }
        parts = [f"[{labels.get(o['agent'], o['agent'])}] {o['output']}" for o in outputs]
        return {"final_answer": "\n\n".join(parts)}

    def _build_graph(self):
        graph = StateGraph(CopilotState)
        graph.add_node("supervisor", self._supervisor_node)
        for agent_name in AGENT_TOOLS:
            graph.add_node(agent_name, self._make_agent_node(agent_name))
        graph.add_node("synthesizer", self._synthesizer_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges("supervisor", self._route_from_supervisor, {a: a for a in AGENT_TOOLS})
        for agent_name in AGENT_TOOLS:
            graph.add_edge(agent_name, "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    def ask(self, question: str) -> CopilotResponse:
        initial_state: CopilotState = {
            "question": question,
            "messages": [],
            "agents_consulted": [],
            "tool_calls_made": [],
            "agent_outputs": [],
            "final_answer": "",
        }
        final_state = self._graph.invoke(initial_state, config={"recursion_limit": 15})

        return CopilotResponse(
            answer=final_state["final_answer"],
            tool_calls=final_state["tool_calls_made"],
            agents_consulted=final_state["agents_consulted"],
            generated_at=dt.datetime.now(dt.UTC),
        )
