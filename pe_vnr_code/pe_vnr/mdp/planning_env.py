from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..planner import MigrationPlan
from ..topology import RiskPrediction, RunningService


@dataclass
class PlanningObservation:
    state: np.ndarray
    action_mask: np.ndarray


class PlanningEnv:
    """Discrete migration-planning MDP adapter."""

    scopes = ("link-only", "partial", "full")

    def __init__(self, max_delay: int = 3):
        self.max_delay = max_delay
        self.action_space_size = 2 * (max_delay + 1) * len(self.scopes)

    def encode_action(self, migrate: int, delay: int, scope_index: int) -> int:
        if migrate not in (0, 1) or not 0 <= delay <= self.max_delay or not 0 <= scope_index < len(self.scopes):
            raise ValueError("Invalid planning action components.")
        return migrate * ((self.max_delay + 1) * len(self.scopes)) + delay * len(self.scopes) + scope_index

    def decode_action(self, action: int) -> Tuple[int, int, int]:
        if not 0 <= action < self.action_space_size:
            raise ValueError("Planning action is outside the action space.")
        migrate_block = (self.max_delay + 1) * len(self.scopes)
        migrate = action // migrate_block
        remain = action % migrate_block
        delay = remain // len(self.scopes)
        scope_index = remain % len(self.scopes)
        return migrate, delay, scope_index

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=bool)
        mask[self.encode_action(0, 0, 0)] = True
        for delay in range(self.max_delay + 1):
            for scope_index in range(len(self.scopes)):
                mask[self.encode_action(1, delay, scope_index)] = True
        return mask

    def build_state(
        self,
        service_features: np.ndarray,
        risk_features: np.ndarray,
        cost_features: np.ndarray,
    ) -> PlanningObservation:
        state = np.concatenate([service_features, risk_features, cost_features]).astype(np.float32)
        return PlanningObservation(state=state, action_mask=self.action_mask())

    def plan_from_action(
        self,
        action: int,
        service: RunningService,
        prediction: RiskPrediction,
    ) -> MigrationPlan:
        migrate, delay, scope_index = self.decode_action(action)
        if migrate == 0:
            return MigrationPlan(False, 0, "none", [], [], 0.0, 0.0)

        scope = self.scopes[scope_index]
        node_targets = [
            v_node_id
            for v_node_id, p_node_id in service.deployment.node_mapping.items()
            if prediction.node_risk.get(p_node_id, 0.0) >= 0.5
        ]
        link_targets = [
            v_link_id
            for v_link_id, path in service.deployment.link_mapping.items()
            if any(prediction.link_risk.get(link_id, 0.0) >= 0.5 for link_id in path)
        ]

        if scope == "link-only":
            node_targets = []
            link_targets = list(service.virtual_graph.edges)
        elif scope == "partial":
            if not node_targets:
                riskiest_v_node = max(
                    service.deployment.node_mapping,
                    key=lambda v_node_id: prediction.node_risk.get(service.deployment.node_mapping[v_node_id], 0.0),
                )
                node_targets = [riskiest_v_node]
            target_set = set(node_targets)
            link_targets = list({*link_targets, *(edge for edge in service.virtual_graph.edges if set(edge) & target_set)})
        else:
            node_targets = list(service.virtual_graph.nodes)
            link_targets = list(service.virtual_graph.edges)

        node_risks = [prediction.node_risk.get(p_node_id, 0.0) for p_node_id in service.deployment.node_mapping.values()]
        link_risks = [
            prediction.link_risk.get(link_id, 0.0)
            for path in service.deployment.link_mapping.values()
            for link_id in path
        ]
        risk_score = max(
            sum(node_risks) / max(1, len(node_risks)),
            sum(link_risks) / max(1, len(link_risks)),
        )
        cost = min(
            1.0,
            0.6 * len(node_targets) / max(1, len(service.deployment.node_mapping))
            + 0.4 * len(link_targets) / max(1, len(service.deployment.link_mapping)),
        )
        return MigrationPlan(True, delay, scope, node_targets, link_targets, cost, risk_score)
