from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from ..planner import MigrationPlan
from ..topology import RunningService


@dataclass
class ExecutionObservation:
    state: np.ndarray
    candidate_nodes: List[int]
    action_mask: np.ndarray


class ExecutionEnv:
    """Sequential node-placement environment for one reconfiguration request."""

    def __init__(self, max_physical_nodes: int = 64):
        self.max_physical_nodes = max_physical_nodes
        self.scope_to_index = {"link-only": 0.0, "partial": 1.0, "full": 2.0}
        self.graph: Optional[nx.Graph] = None
        self.service: Optional[RunningService] = None
        self.plan: Optional[MigrationPlan] = None
        self.node_risk: Dict[int, float] = {}
        self.node_mapping: Dict[int, int] = {}
        self.pending_v_nodes: List[int] = []
        self.cursor = 0
        self.used_nodes: Set[int] = set()

    def _reserve_node(self, node_id: int, demand: float) -> None:
        assert self.graph is not None
        self.graph.nodes[node_id]["cpu"] = float(self.graph.nodes[node_id].get("cpu", 0.0)) - demand

    def _candidate_nodes(self, v_node_id: int) -> List[int]:
        assert self.graph is not None and self.service is not None
        demand = float(self.service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
        return sorted(
            [
                node_id
                for node_id, attrs in self.graph.nodes(data=True)
                if node_id not in self.used_nodes and float(attrs.get("cpu", 0.0)) >= demand
            ],
            key=lambda node_id: (
                self.node_risk.get(node_id, 0.0),
                -float(self.graph.nodes[node_id].get("cpu", 0.0)),
                float(self.graph.nodes[node_id].get("queue", 0.0)),
            ),
        )

    def reset(
        self,
        graph: nx.Graph,
        service: RunningService,
        plan: MigrationPlan,
        node_risk: Dict[int, float],
    ) -> Optional[ExecutionObservation]:
        self.graph = nx.Graph(graph)
        self.service = service
        self.plan = plan
        self.node_risk = dict(node_risk)
        self.node_mapping = {}
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
        return self.current_observation()

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
    ) -> ExecutionObservation:
        scope_value = np.array([self.scope_to_index.get(scope, 0.0)], dtype=np.float32)
        state = np.concatenate([current_v_node_features, graph_features, risk_features, scope_value]).astype(np.float32)
        mask = np.zeros(self.max_physical_nodes, dtype=bool)
        for node_id in candidate_nodes:
            if 0 <= node_id < self.max_physical_nodes:
                mask[node_id] = True
        return ExecutionObservation(state=state, candidate_nodes=candidate_nodes, action_mask=mask)

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
            [self.node_risk.get(node_id, 0.0) for node_id in candidates[:3]] + [0.0] * max(0, 3 - len(candidates)),
            dtype=np.float32,
        )
        v_features = np.array(
            [demand / 100.0, self.service.virtual_graph.degree[v_node_id] / max(1, self.service.virtual_graph.number_of_nodes() - 1)],
            dtype=np.float32,
        )
        return self.build_state(v_features, graph_features, risk_features, self.plan.scope, candidates)

    def step(self, action: int) -> Tuple[Optional[ExecutionObservation], bool, Dict[str, object]]:
        observation = self.current_observation()
        if observation is None:
            raise RuntimeError("Execution episode has already terminated.")
        if action not in observation.candidate_nodes:
            raise ValueError("Selected physical node is not a feasible action.")
        assert self.service is not None
        v_node_id = self.pending_v_nodes[self.cursor]
        demand = float(self.service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
        self.node_mapping[v_node_id] = action
        self._reserve_node(action, demand)
        self.used_nodes.add(action)
        self.cursor += 1
        return self.current_observation(), self.done, {"virtual_node": v_node_id, "physical_node": action}
