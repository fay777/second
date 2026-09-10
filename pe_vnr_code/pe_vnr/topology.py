from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx


NodeId = int
LinkId = Tuple[int, int]
VNodeId = int
VLinkId = Tuple[int, int]


@dataclass
class NodeResource:
    cpu: float
    storage: float = 0.0
    energy: float = 1.0
    queue: float = 0.0
    domain: int = 0
    max_cpu: float = 100.0


@dataclass
class LinkResource:
    bandwidth: float
    delay: float
    loss: float = 0.0
    visible_time: float = 1.0
    max_bandwidth: float = 100.0


@dataclass
class Deployment:
    node_mapping: Dict[VNodeId, NodeId] = field(default_factory=dict)
    link_mapping: Dict[VLinkId, List[LinkId]] = field(default_factory=dict)
    description: str = ""

    def clone(self) -> "Deployment":
        return Deployment(
            node_mapping=dict(self.node_mapping),
            link_mapping={k: list(v) for k, v in self.link_mapping.items()},
            description=self.description,
        )


@dataclass
class RunningService:
    service_id: str
    virtual_graph: nx.Graph
    deployment: Deployment
    remaining_lifetime: int
    max_delay: float
    priority: float = 1.0
    migration_count: int = 0
    migrated_virtual_nodes: int = 0
    rerouted_virtual_links: int = 0
    disruption_time: float = 0.0
    sla_violations: int = 0
    status: str = "running"

    def clone(self) -> "RunningService":
        return RunningService(
            service_id=self.service_id,
            virtual_graph=graph_copy(self.virtual_graph),
            deployment=self.deployment.clone(),
            remaining_lifetime=self.remaining_lifetime,
            max_delay=self.max_delay,
            priority=self.priority,
            migration_count=self.migration_count,
            migrated_virtual_nodes=self.migrated_virtual_nodes,
            rerouted_virtual_links=self.rerouted_virtual_links,
            disruption_time=self.disruption_time,
            sla_violations=self.sla_violations,
            status=self.status,
        )


@dataclass
class RiskPrediction:
    node_risk: Dict[NodeId, float]
    link_risk: Dict[LinkId, float]
    future_node_risk: List[Dict[NodeId, float]]
    future_link_risk: List[Dict[LinkId, float]]

    @property
    def mean_node_risk(self) -> float:
        if not self.node_risk:
            return 0.0
        return sum(self.node_risk.values()) / len(self.node_risk)

    @property
    def mean_link_risk(self) -> float:
        if not self.link_risk:
            return 0.0
        return sum(self.link_risk.values()) / len(self.link_risk)

    @property
    def max_node_risk(self) -> float:
        if not self.node_risk:
            return 0.0
        return max(self.node_risk.values())

    @property
    def max_link_risk(self) -> float:
        if not self.link_risk:
            return 0.0
        return max(self.link_risk.values())


@dataclass
class TimeVaryingTopology:
    history: List[nx.Graph]

    @property
    def current(self) -> nx.Graph:
        return self.history[-1]

    def window(self, size: int) -> List[nx.Graph]:
        if size <= 0:
            return []
        return self.history[-size:]


def edge_key(u: int, v: int) -> LinkId:
    return (u, v) if u <= v else (v, u)


def graph_copy(graph: nx.Graph) -> nx.Graph:
    return nx.Graph(graph)
