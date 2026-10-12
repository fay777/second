from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from ..planner import MigrationPlan
from ..topology import RunningService, edge_key


@dataclass
class ExecutionObservation:
    state: np.ndarray
    candidate_nodes: List[int]
    candidate_features: np.ndarray


class ExecutionEnv:
    """Sequential node-placement environment for one reconfiguration request."""

    def __init__(self):
        self.scope_to_index = {"link-only": 0.0, "partial": 1.0, "full": 2.0}
        self.graph: Optional[nx.Graph] = None
        self.service: Optional[RunningService] = None
        self.plan: Optional[MigrationPlan] = None
        self.node_risk: Dict[int, float] = {}
        self.link_risk: Dict[Tuple[int, int], float] = {}
        self.future_node_risk: List[Dict[int, float]] = []
        self.node_mapping: Dict[int, int] = {}
        self.link_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.pending_v_nodes: List[int] = []
        self.cursor = 0
        self.used_nodes: Set[int] = set()
        self.routing_risk_weight = 1.0

    def _reserve_node(self, node_id: int, demand: float) -> None:
        assert self.graph is not None
        self.graph.nodes[node_id]["cpu"] = float(self.graph.nodes[node_id].get("cpu", 0.0)) - demand

    @staticmethod
    def _reserve_path(graph: nx.Graph, path: List[Tuple[int, int]], demand: float) -> None:
        for link_id in path:
            edge = edge_key(*link_id)
            graph.edges[edge]["bandwidth"] = float(graph.edges[edge].get("bandwidth", 0.0)) - demand

    def _find_best_path(self, graph: nx.Graph, source: int, target: int, demand: float) -> List[Tuple[int, int]]:
        """Mirror the executor's risk-aware feasible routing on this local graph."""
        feasible = nx.Graph()
        feasible.add_nodes_from(
            (node_id, attrs)
            for node_id, attrs in graph.nodes(data=True)
            if float(attrs.get("fault", 0.0)) < 1.0 and float(attrs.get("available", 1.0)) > 0.0
        )
        for u, v, attrs in graph.edges(data=True):
            if (
                u not in feasible or v not in feasible
                or float(attrs.get("fault", 0.0)) >= 1.0
                or float(attrs.get("bandwidth", 0.0)) < demand
                or float(attrs.get("visible_time", 1.0)) <= 0.0
            ):
                continue
            weight = (
                self.routing_risk_weight * self.link_risk.get(edge_key(u, v), 0.0)
                + 0.05 * float(attrs.get("delay", 0.0))
                + 0.02 * float(attrs.get("loss", 0.0))
            )
            feasible.add_edge(u, v, routing_weight=weight)
        try:
            nodes = nx.shortest_path(feasible, source, target, weight="routing_weight", method="dijkstra")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return [edge_key(u, v) for u, v in zip(nodes[:-1], nodes[1:])]

    def _route_completed_links(
        self,
        graph: nx.Graph,
        node_mapping: Dict[int, int],
        link_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]],
        virtual_node: int,
    ) -> bool:
        """Reserve all rerouted virtual links made complete by one placement."""
        assert self.service is not None and self.plan is not None
        reroute_links = set(self.plan.target_v_links)
        for neighbor in self.service.virtual_graph.neighbors(virtual_node):
            v_link = edge_key(virtual_node, neighbor)
            if neighbor not in node_mapping or v_link in link_mapping:
                continue
            if v_link not in reroute_links:
                return False
            demand = float(self.service.virtual_graph.edges[virtual_node, neighbor].get("bandwidth", 0.0))
            path = self._find_best_path(graph, node_mapping[virtual_node], node_mapping[neighbor], demand)
            if not path:
                return False
            link_mapping[v_link] = path
            self._reserve_path(graph, path, demand)
        return True

    def _raw_candidate_nodes(self, v_node_id: int) -> List[int]:
        """Return hosts feasible for one VNF before remaining-placement lookahead."""
        assert self.graph is not None and self.service is not None
        demand = float(self.service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
        return sorted(
            [
                node_id
                for node_id, attrs in self.graph.nodes(data=True)
                if (
                    node_id not in self.used_nodes
                    and float(attrs.get("cpu", 0.0)) >= demand
                    and float(attrs.get("fault", 0.0)) < 1.0
                    and float(attrs.get("available", 1.0)) > 0.0
                    and float(attrs.get("energy", 1.0)) > 0.05
                    and node_id != self.service.deployment.node_mapping.get(v_node_id)
                    and self._mapped_neighbor_paths_feasible(v_node_id, node_id)
                )
            ],
            key=lambda node_id: (
                self._risk_exposure(node_id),
                -float(self.graph.nodes[node_id].get("cpu", 0.0)),
                float(self.graph.nodes[node_id].get("queue", 0.0)),
            ),
        )

    def _node_resource_candidates(self, v_node_id: int) -> List[int]:
        """Return CPU/node-state candidates without repeatedly constructing routes."""
        assert self.graph is not None and self.service is not None
        demand = float(self.service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
        return [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if (
                node_id not in self.used_nodes
                and node_id != self.service.deployment.node_mapping.get(v_node_id)
                and float(attrs.get("cpu", 0.0)) >= demand
                and float(attrs.get("fault", 0.0)) < 1.0
                and float(attrs.get("available", 1.0)) > 0.0
                and float(attrs.get("energy", 1.0)) > 0.05
            )
        ]

    def _remaining_targets_matchable(self, candidate_node: int) -> bool:
        """Reject a host that violates a necessary remaining CPU-host matching.

        The lookahead intentionally ignores routing. Routing remains a
        deterministic executor operation, while this inexpensive necessary
        condition prevents the chosen host from consuming the sole CPU-feasible
        host of a later VNF.
        """
        assert self.graph is not None
        if self.cursor + 1 >= len(self.pending_v_nodes):
            return True

        target_nodes = self.pending_v_nodes[self.cursor + 1:]
        bipartite_graph = nx.Graph()
        left_nodes = [("virtual", target) for target in target_nodes]
        bipartite_graph.add_nodes_from(left_nodes, bipartite=0)
        for target in target_nodes:
            physical_candidates = [
                node_id for node_id in self._node_resource_candidates(target)
                if node_id != candidate_node
            ]
            if not physical_candidates:
                return False
            virtual_key = ("virtual", target)
            for physical_node in physical_candidates:
                bipartite_graph.add_edge(virtual_key, ("physical", physical_node))
        matching = nx.algorithms.bipartite.maximum_matching(
            bipartite_graph, top_nodes=left_nodes
        )
        return sum(virtual_key in matching for virtual_key in left_nodes) == len(left_nodes)

    def _candidate_nodes(self, v_node_id: int) -> List[int]:
        candidates = self._raw_candidate_nodes(v_node_id)
        candidates = [
            node_id
            for node_id in candidates
            if self._remaining_targets_matchable(node_id)
        ]
        return candidates

    def _mapped_neighbor_paths_feasible(self, v_node_id: int, candidate_node: int) -> bool:
        """Filter candidates that cannot reserve newly completed virtual links."""
        assert self.graph is not None and self.service is not None
        candidate_graph = nx.Graph(self.graph)
        candidate_mapping = dict(self.node_mapping)
        candidate_mapping[v_node_id] = candidate_node
        candidate_links = {link: list(path) for link, path in self.link_mapping.items()}
        return self._route_completed_links(
            candidate_graph, candidate_mapping, candidate_links, v_node_id
        )

    def _risk_exposure(self, node_id: int) -> float:
        future = [risk.get(node_id, 0.0) for risk in self.future_node_risk]
        if future:
            return sum(future) / len(future)
        return self.node_risk.get(node_id, 0.0)

    def reset(
        self,
        graph: nx.Graph,
        service: RunningService,
        plan: MigrationPlan,
        node_risk: Dict[int, float],
        future_node_risk: Optional[List[Dict[int, float]]] = None,
        link_risk: Optional[Dict[Tuple[int, int], float]] = None,
        routing_risk_weight: float = 1.0,
    ) -> Optional[ExecutionObservation]:
        self.graph = nx.Graph(graph)
        self.service = service
        self.plan = plan
        self.node_risk = dict(node_risk)
        self.link_risk = dict(link_risk or {})
        self.future_node_risk = list(future_node_risk or [])
        self.routing_risk_weight = float(routing_risk_weight)
        self.node_mapping = {}
        self.link_mapping = {}
        self.used_nodes = set()
        self.cursor = 0
        targets = set(plan.target_v_nodes)
        self.pending_v_nodes = [v_node_id for v_node_id in service.virtual_graph.nodes if v_node_id in targets]
        for v_node_id, p_node_id in service.deployment.node_mapping.items():
            if v_node_id in targets:
                continue
            demand = float(service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
            if float(self.graph.nodes[p_node_id].get("cpu", 0.0)) < demand:
                raise RuntimeError(f"Fixed node {p_node_id} cannot host virtual node {v_node_id}.")
            self.node_mapping[v_node_id] = p_node_id
            self._reserve_node(p_node_id, demand)
            self.used_nodes.add(p_node_id)
        # The service is released before this worker runs. Re-reserve all
        # unchanged virtual links so Lower observes their real BW footprint.
        reroute_links = set(plan.target_v_links)
        for u, v, attrs in service.virtual_graph.edges(data=True):
            v_link = edge_key(u, v)
            if v_link in reroute_links:
                continue
            if u not in self.node_mapping or v not in self.node_mapping:
                raise RuntimeError(f"Unmapped preserved virtual link {v_link}.")
            path = list(service.deployment.link_mapping.get(v_link, []))
            demand = float(attrs.get("bandwidth", 0.0))
            if not path or any(
                not self.graph.has_edge(*edge_key(*link_id))
                or float(self.graph.edges[edge_key(*link_id)].get("bandwidth", 0.0)) < demand
                for link_id in path
            ):
                raise RuntimeError(f"Preserved path for virtual link {v_link} is no longer feasible.")
            self.link_mapping[v_link] = path
            self._reserve_path(self.graph, path, demand)
        return self.current_observation()

    @classmethod
    def can_start(
        cls,
        graph: nx.Graph,
        service: RunningService,
        plan: MigrationPlan,
        node_risk: Dict[int, float],
        future_node_risk: Optional[List[Dict[int, float]]] = None,
        link_risk: Optional[Dict[Tuple[int, int], float]] = None,
        routing_risk_weight: float = 1.0,
    ) -> bool:
        """Check whether a node-remapping request can begin without mutating its owner."""
        try:
            return cls().reset(
                graph, service, plan, node_risk, future_node_risk,
                link_risk, routing_risk_weight,
            ) is not None
        except (RuntimeError, ValueError, nx.NetworkXError):
            return False

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.pending_v_nodes)

    def build_state(
        self,
        current_v_node_features: np.ndarray,
        graph_features: np.ndarray,
        risk_features: np.ndarray,
        scope: str,
        candidate_nodes: List[int],
        candidate_features: np.ndarray,
    ) -> ExecutionObservation:
        scope_value = np.array([self.scope_to_index.get(scope, 0.0)], dtype=np.float32)
        state = np.concatenate([current_v_node_features, graph_features, risk_features, scope_value]).astype(np.float32)
        return ExecutionObservation(state=state, candidate_nodes=candidate_nodes, candidate_features=candidate_features)

    def current_observation(self) -> Optional[ExecutionObservation]:
        if self.done:
            return None
        assert self.graph is not None and self.service is not None and self.plan is not None
        v_node_id = self.pending_v_nodes[self.cursor]
        candidates = self._candidate_nodes(v_node_id)
        if not candidates:
            raise RuntimeError(f"No feasible host for virtual node {v_node_id}.")
        demand = float(self.service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
        graph_features = np.array(
            [
                np.mean([float(attrs.get("cpu", 0.0)) / max(float(attrs.get("max_cpu", 1.0)), 1e-6) for _, attrs in self.graph.nodes(data=True)]),
                np.mean([float(attrs.get("queue", 0.0)) for _, attrs in self.graph.nodes(data=True)]),
                len(candidates) / max(1, self.graph.number_of_nodes()),
            ],
            dtype=np.float32,
        )
        risk_features = np.array(
            [self._risk_exposure(node_id) for node_id in candidates[:3]] + [0.0] * max(0, 3 - len(candidates)),
            dtype=np.float32,
        )
        v_features = np.array(
            [demand / 100.0, self.service.virtual_graph.degree[v_node_id] / max(1, self.service.virtual_graph.number_of_nodes() - 1)],
            dtype=np.float32,
        )
        candidate_features = np.array(
            [
                [
                    float(self.graph.nodes[node_id].get("cpu", 0.0)) / max(float(self.graph.nodes[node_id].get("max_cpu", 1.0)), 1e-6),
                    float(self.graph.nodes[node_id].get("queue", 0.0)),
                    float(self.graph.nodes[node_id].get("energy", 1.0)),
                    float(self.graph.nodes[node_id].get("domain", 0.0)) / 2.0,
                    self.node_risk.get(node_id, 0.0),
                    self._risk_exposure(node_id),
                    self.graph.degree[node_id] / max(1, self.graph.number_of_nodes() - 1),
                ]
                for node_id in candidates
            ],
            dtype=np.float32,
        )
        return self.build_state(v_features, graph_features, risk_features, self.plan.scope, candidates, candidate_features)

    def step(self, action: int) -> Tuple[Optional[ExecutionObservation], bool, Dict[str, object]]:
        observation = self.current_observation()
        if observation is None:
            raise RuntimeError("Execution episode has already terminated.")
        if not 0 <= action < len(observation.candidate_nodes):
            raise ValueError("Selected candidate index is not a feasible action.")
        assert self.service is not None
        v_node_id = self.pending_v_nodes[self.cursor]
        demand = float(self.service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
        physical_node = observation.candidate_nodes[action]
        self.node_mapping[v_node_id] = physical_node
        self._reserve_node(physical_node, demand)
        self.used_nodes.add(physical_node)
        if not self._route_completed_links(
            self.graph, self.node_mapping, self.link_mapping, v_node_id
        ):
            raise RuntimeError(f"No feasible physical path for virtual node {v_node_id}.")
        self.cursor += 1
        return self.current_observation(), self.done, {"virtual_node": v_node_id, "physical_node": physical_node}
