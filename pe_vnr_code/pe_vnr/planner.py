from dataclasses import dataclass
from typing import Dict, List

from .config import PlanningConfig
from .topology import RiskPrediction, RunningService


@dataclass
class MigrationPlan:
    migrate: bool
    trigger_delay: int
    scope: str
    target_v_nodes: List[int]
    target_v_links: List[tuple]
    estimated_cost: float
    risk_score: float


class MigrationPlanner:
    def __init__(self, config: PlanningConfig):
        self.config = config

    def _service_risk(self, service: RunningService, prediction: RiskPrediction) -> Dict[str, float]:
        node_risks = []
        for _, p_node_id in service.deployment.node_mapping.items():
            node_risks.append(prediction.node_risk.get(p_node_id, 0.0))

        link_risks = []
        for _, physical_path in service.deployment.link_mapping.items():
            for link_id in physical_path:
                link_risks.append(prediction.link_risk.get(link_id, 0.0))

        mean_node = sum(node_risks) / max(1, len(node_risks))
        mean_link = sum(link_risks) / max(1, len(link_risks))
        return {
            "mean_node": mean_node,
            "mean_link": mean_link,
            "risk_score": max(mean_node, mean_link),
        }

    def _estimate_cost(self, service: RunningService, node_targets: List[int], link_targets: List[tuple], scope: str) -> float:
        total_nodes = max(1, len(service.deployment.node_mapping))
        total_links = max(1, len(service.deployment.link_mapping))
        if scope == "none":
            return 0.0
        return min(
            1.0,
            0.6 * len(node_targets) / total_nodes + 0.4 * len(link_targets) / total_links,
        )

    def plan(self, service: RunningService, prediction: RiskPrediction) -> MigrationPlan:
        service_risk = self._service_risk(service, prediction)
        node_targets = []
        for v_node_id, p_node_id in service.deployment.node_mapping.items():
            if prediction.node_risk.get(p_node_id, 0.0) >= self.config.node_risk_threshold:
                node_targets.append(v_node_id)

        link_targets = []
        for v_link_id, physical_path in service.deployment.link_mapping.items():
            if any(prediction.link_risk.get(link_id, 0.0) >= self.config.link_risk_threshold for link_id in physical_path):
                link_targets.append(v_link_id)

        risky_ratio = len(node_targets) / max(1, len(service.deployment.node_mapping))
        risk_score = service_risk["risk_score"]
        if risk_score < min(self.config.node_risk_threshold, self.config.link_risk_threshold):
            scope = "none"
        elif not node_targets and link_targets:
            scope = "link-only"
        elif risk_score >= self.config.full_migration_threshold or risky_ratio >= self.config.partial_ratio_threshold:
            scope = "full"
            node_targets = list(service.virtual_graph.nodes)
            link_targets = list(service.virtual_graph.edges)
        else:
            scope = "partial"
            target_nodes = set(node_targets)
            # A migrated endpoint invalidates its incident virtual-link path.
            link_targets = [
                v_link_id
                for v_link_id in service.virtual_graph.edges
                if set(v_link_id) & target_nodes
            ]

        migrate = scope != "none"
        if not migrate:
            delay = 0
        elif risk_score >= self.config.full_migration_threshold:
            delay = 0
        else:
            delay = min(self.config.max_trigger_delay, int((1.0 - risk_score) * self.config.max_trigger_delay))

        cost = self._estimate_cost(service, node_targets, link_targets, scope)
        return MigrationPlan(
            migrate=migrate,
            trigger_delay=delay,
            scope=scope,
            target_v_nodes=node_targets,
            target_v_links=link_targets,
            estimated_cost=cost,
            risk_score=risk_score,
        )
