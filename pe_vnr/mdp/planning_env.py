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

    def __init__(self, max_delay: int = 3, node_risk_threshold: float = 0.5, link_risk_threshold: float = 0.5):
        self.max_delay = max_delay
        self.node_risk_threshold = node_risk_threshold
        self.link_risk_threshold = link_risk_threshold
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

    def _risk_targets(
        self,
        service: RunningService,
        prediction: RiskPrediction,
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        future_node_risk = prediction.future_node_risk or [prediction.node_risk]
        future_link_risk = prediction.future_link_risk or [prediction.link_risk]
        node_targets = [
            v_node_id
            for v_node_id, p_node_id in service.deployment.node_mapping.items()
            if max(risk.get(p_node_id, 0.0) for risk in future_node_risk) >= self.node_risk_threshold
        ]
        link_targets = [
            v_link_id
            for v_link_id, path in service.deployment.link_mapping.items()
            if any(max(risk.get(link_id, 0.0) for risk in future_link_risk) >= self.link_risk_threshold for link_id in path)
        ]
        return node_targets, link_targets

    def action_mask(
        self,
        service: RunningService = None,
        prediction: RiskPrediction = None,
    ) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=bool)
        mask[self.encode_action(0, 0, 0)] = True
        node_targets, link_targets = ([], [])
        if service is not None and prediction is not None:
            node_targets, link_targets = self._risk_targets(service, prediction)
        for delay in range(self.max_delay + 1):
            if service is not None and delay >= service.remaining_lifetime:
                continue
            for scope_index in range(len(self.scopes)):
                scope = self.scopes[scope_index]
                # Do not sample actions that would be silently converted to KEEP.
                if scope == "link-only" and not link_targets:
                    continue
                if scope == "partial" and not node_targets:
                    continue
                mask[self.encode_action(1, delay, scope_index)] = True
        return mask

    def build_state(
        self,
        service_features: np.ndarray,
        risk_features: np.ndarray,
        cost_features: np.ndarray,
        service: RunningService = None,
        prediction: RiskPrediction = None,
    ) -> PlanningObservation:
        state = np.concatenate([service_features, risk_features, cost_features]).astype(np.float32)
        return PlanningObservation(state=state, action_mask=self.action_mask(service, prediction))

    def plan_from_action(
        self,
        action: int,
        service: RunningService,
        prediction: RiskPrediction,
    ) -> MigrationPlan:
        future_node_risk = prediction.future_node_risk or [prediction.node_risk]
        future_link_risk = prediction.future_link_risk or [prediction.link_risk]
        expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
        node_risks = [expected_node_risk.get(p_node_id, 0.0) for p_node_id in service.deployment.node_mapping.values()]
        link_risks = [expected_link_risk.get(link_id, 0.0) for path in service.deployment.link_mapping.values() for link_id in path]
        risk_score = max(
            sum(node_risks) / max(1, len(node_risks)),
            sum(link_risks) / max(1, len(link_risks)),
        )
        migrate, delay, scope_index = self.decode_action(action)
        if migrate == 0:
            return MigrationPlan(False, 0, "none", [], [], 0.0, risk_score)

        scope = self.scopes[scope_index]
        node_targets, link_targets = self._risk_targets(service, prediction)

        if scope == "link-only":
            node_targets = []
            if not link_targets:
                return MigrationPlan(False, 0, "none", [], [], 0.0, risk_score)
        elif scope == "partial":
            if not node_targets:
                if link_targets:
                    scope = "link-only"
                else:
                    return MigrationPlan(False, 0, "none", [], [], 0.0, risk_score)
            target_set = set(node_targets)
            link_targets = list({*link_targets, *(edge for edge in service.virtual_graph.edges if set(edge) & target_set)})
        else:
            node_targets = list(service.virtual_graph.nodes)
            link_targets = list(service.virtual_graph.edges)

        cost = min(
            1.0,
            0.6 * len(node_targets) / max(1, len(service.deployment.node_mapping))
            + 0.4 * len(link_targets) / max(1, len(service.deployment.link_mapping)),
        )
        return MigrationPlan(True, delay, scope, node_targets, link_targets, cost, risk_score)
