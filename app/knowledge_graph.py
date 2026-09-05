from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import networkx as nx

from app.config import settings


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    attributes: dict


@dataclass
class GraphEdge:
    source: str
    target: str
    relationship: str
    attributes: dict


class Neo4jGateway:
    def __init__(self) -> None:
        self._driver = None
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                settings.neo4j.uri, auth=(settings.neo4j.user, settings.neo4j.password)
            )
            self._driver.verify_connectivity()
        except Exception:
            self._driver = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def upsert_node(self, node: GraphNode) -> None:
        if not self._driver:
            return
        query = f"MERGE (n:{node.node_type} {{id: $id}}) SET n += $attrs"
        with self._driver.session() as session:
            session.run(query, id=node.node_id, attrs=node.attributes)

    def upsert_edge(self, edge: GraphEdge) -> None:
        if not self._driver:
            return
        query = (
            "MATCH (a {id: $source}), (b {id: $target}) "
            f"MERGE (a)-[r:{edge.relationship}]->(b) SET r += $attrs"
        )
        with self._driver.session() as session:
            session.run(query, source=edge.source, target=edge.target, attrs=edge.attributes)

    def close(self) -> None:
        if self._driver:
            self._driver.close()


class LegalKnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.neo4j = Neo4jGateway()

    def add_company(self, company_id: str, name: str) -> None:
        self.graph.add_node(company_id, node_type="Company", name=name)
        self.neo4j.upsert_node(GraphNode(company_id, "Company", {"name": name}))

    def add_jurisdiction(self, jurisdiction_id: str, name: str) -> None:
        self.graph.add_node(jurisdiction_id, node_type="Jurisdiction", name=name)
        self.neo4j.upsert_node(GraphNode(jurisdiction_id, "Jurisdiction", {"name": name}))

    def add_contract(self, contract_id: str, document_type: str, jurisdiction: str, value: float) -> None:
        self.graph.add_node(
            contract_id, node_type="Contract", document_type=document_type, jurisdiction=jurisdiction, value=value
        )
        self.neo4j.upsert_node(
            GraphNode(contract_id, "Contract", {"document_type": document_type, "jurisdiction": jurisdiction, "value": value})
        )

    def add_vendor(self, vendor_id: str, name: str, risk_score: float = 0.0) -> None:
        self.graph.add_node(vendor_id, node_type="Vendor", name=name, risk_score=risk_score)
        self.neo4j.upsert_node(GraphNode(vendor_id, "Vendor", {"name": name, "risk_score": risk_score}))

    def add_employee(self, employee_id: str, name: str, department: str) -> None:
        self.graph.add_node(employee_id, node_type="Employee", name=name, department=department)
        self.neo4j.upsert_node(GraphNode(employee_id, "Employee", {"name": name, "department": department}))

    def add_policy(self, policy_id: str, name: str, framework: str) -> None:
        self.graph.add_node(policy_id, node_type="Policy", name=name, framework=framework)
        self.neo4j.upsert_node(GraphNode(policy_id, "Policy", {"name": name, "framework": framework}))

    def add_clause_node(self, clause_id: str, clause_type: str, contract_id: str) -> None:
        self.graph.add_node(clause_id, node_type="Clause", clause_type=clause_type)
        self.graph.add_edge(contract_id, clause_id, relationship="CONTAINS")
        self.neo4j.upsert_node(GraphNode(clause_id, "Clause", {"clause_type": clause_type}))
        self.neo4j.upsert_edge(GraphEdge(contract_id, clause_id, "CONTAINS", {}))

    def link_company_to_contract(self, company_id: str, contract_id: str) -> None:
        self.graph.add_edge(company_id, contract_id, relationship="PARTY_TO")
        self.neo4j.upsert_edge(GraphEdge(company_id, contract_id, "PARTY_TO", {}))

    def link_vendor_to_contract(self, vendor_id: str, contract_id: str) -> None:
        self.graph.add_edge(vendor_id, contract_id, relationship="COUNTERPARTY")
        self.neo4j.upsert_edge(GraphEdge(vendor_id, contract_id, "COUNTERPARTY", {}))

    def link_employee_to_contract(self, employee_id: str, contract_id: str, role: str = "owner") -> None:
        self.graph.add_edge(employee_id, contract_id, relationship="MANAGES", role=role)
        self.neo4j.upsert_edge(GraphEdge(employee_id, contract_id, "MANAGES", {"role": role}))

    def link_contract_to_policy(self, contract_id: str, policy_id: str) -> None:
        self.graph.add_edge(contract_id, policy_id, relationship="SUBJECT_TO")
        self.neo4j.upsert_edge(GraphEdge(contract_id, policy_id, "SUBJECT_TO", {}))

    def link_contract_to_jurisdiction(self, contract_id: str, jurisdiction_id: str) -> None:
        self.graph.add_edge(contract_id, jurisdiction_id, relationship="GOVERNED_IN")
        self.neo4j.upsert_edge(GraphEdge(contract_id, jurisdiction_id, "GOVERNED_IN", {}))

    def record_obligation_link(self, obligation_id: str, contract_id: str, due_date: str) -> None:
        self.graph.add_node(obligation_id, node_type="Obligation", due_date=due_date)
        self.graph.add_edge(contract_id, obligation_id, relationship="REQUIRES")
        self.neo4j.upsert_node(GraphNode(obligation_id, "Obligation", {"due_date": due_date}))
        self.neo4j.upsert_edge(GraphEdge(contract_id, obligation_id, "REQUIRES", {}))

    def find_related_contracts(self, vendor_id: str) -> list[str]:
        if vendor_id not in self.graph:
            return []
        return [
            n for n in self.graph.successors(vendor_id)
            if self.graph.nodes[n].get("node_type") == "Contract"
        ]

    def find_high_exposure_vendors(self, top_k: int = 10) -> list[dict]:
        vendor_exposure: dict[str, float] = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "Vendor":
                continue
            contracts = self.find_related_contracts(node)
            total_value = sum(self.graph.nodes[c].get("value", 0.0) for c in contracts)
            vendor_exposure[node] = total_value
        ranked = sorted(vendor_exposure.items(), key=lambda kv: kv[1], reverse=True)
        return [{"vendor": v, "total_exposure": round(val, 2)} for v, val in ranked[:top_k]]

    def find_contracts_by_jurisdiction(self, jurisdiction_id: str) -> list[str]:
        if jurisdiction_id not in self.graph:
            return []
        return [
            n for n in self.graph.predecessors(jurisdiction_id)
            if self.graph.nodes[n].get("node_type") == "Contract"
        ]

    def find_shortest_path(self, source: str, target: str) -> list[str]:
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def contract_criticality_ranking(self) -> list[dict]:
        centrality = nx.betweenness_centrality(self.graph.to_undirected(), k=min(200, len(self.graph)))
        ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"node": node, "criticality_score": round(score, 5), "node_type": self.graph.nodes[node].get("node_type")}
            for node, score in ranked[:20]
            if self.graph.nodes[node].get("node_type") == "Contract"
        ]

    def summary(self) -> dict:
        node_type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_type_counts[data.get("node_type", "unknown")] = node_type_counts.get(data.get("node_type", "unknown"), 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_type_breakdown": node_type_counts,
            "neo4j_connected": self.neo4j.is_connected,
        }

    def seed_sample_graph(self, num_contracts: int = 50) -> dict:
        from app.document_pipeline import LegalDocumentSimulator

        simulator = LegalDocumentSimulator()
        documents = simulator.generate_batch(num_contracts)

        self.add_company("COMPANY-HQ", "Global Enterprise Holdings")
        policy_ids = []
        for i, framework in enumerate(["GDPR", "SOX", "Procurement Standard", "Data Privacy Policy"]):
            policy_id = f"POLICY-{i}"
            self.add_policy(policy_id, framework, framework)
            policy_ids.append(policy_id)

        jurisdiction_ids: dict[str, str] = {}
        for i, doc in enumerate(documents):
            contract_id = doc.document_code
            self.add_contract(contract_id, doc.document_type, doc.jurisdiction, doc.contract_value)
            self.link_company_to_contract("COMPANY-HQ", contract_id)

            if doc.jurisdiction not in jurisdiction_ids:
                jurisdiction_id = f"JURISDICTION-{len(jurisdiction_ids)}"
                self.add_jurisdiction(jurisdiction_id, doc.jurisdiction)
                jurisdiction_ids[doc.jurisdiction] = jurisdiction_id
            self.link_contract_to_jurisdiction(contract_id, jurisdiction_ids[doc.jurisdiction])

            vendor_id = f"VENDOR-{i % 20}"
            self.add_vendor(vendor_id, doc.counterparty, risk_score=round((i % 10) / 10, 2))
            self.link_vendor_to_contract(vendor_id, contract_id)

            employee_id = f"EMP-{i % 12}"
            self.add_employee(employee_id, f"Legal Officer {i % 12}", doc.department)
            self.link_employee_to_contract(employee_id, contract_id)

            self.link_contract_to_policy(contract_id, policy_ids[i % len(policy_ids)])

        return self.summary()
