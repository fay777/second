from dataclasses import dataclass


@dataclass
class MigrationPlan:
    migrate: bool
    trigger_delay: int
    scope: str
    target_v_nodes: list
    target_v_links: list
    estimated_cost: float
    risk_score: float


class MigrationPlanner:
    def __init__(
        self,
        node_risk_threshold=0.65,
        link_risk_threshold=0.70,
        full_migration_threshold=0.80,
        partial_ratio_threshold=0.35,
        max_trigger_delay=3,
    ):
        self.node_risk_threshold = node_risk_threshold
        self.link_risk_threshold = link_risk_threshold
        self.full_migration_threshold = full_migration_threshold
        self.partial_ratio_threshold = partial_ratio_threshold
        self.max_trigger_delay = max_trigger_delay

    def _estimate_migration_cost(self, current_solution, target_v_nodes, target_v_links):
        if current_solution is None:
            return 0.0
        total_elements = max(1, len(current_solution["node_slots"]) + len(current_solution["link_paths"]))
        moved_elements = len(target_v_nodes) + len(target_v_links)
        return float(moved_elements / total_elements)

    def plan(self, v_net, current_solution, risk_prediction):
        if current_solution is None or len(current_solution["node_slots"]) == 0:
            return MigrationPlan(
                migrate=False,
                trigger_delay=0,
                scope="none",
                target_v_nodes=[],
                target_v_links=[],
                estimated_cost=0.0,
                risk_score=0.0,
            )

        node_slots = current_solution["node_slots"]
        target_v_nodes = [
            v_node_id
            for v_node_id, p_node_id in node_slots.items()
            if risk_prediction.node_risk.get(p_node_id, 0.0) >= self.node_risk_threshold
        ]

        target_v_links = []
        for v_link, p_links in current_solution["link_paths"].items():
            if any(risk_prediction.link_risk.get(p_link, 0.0) >= self.link_risk_threshold for p_link in p_links):
                target_v_links.append(v_link)

        risky_nodes_ratio = len(target_v_nodes) / max(1, len(node_slots))
        risk_score = max(risk_prediction.max_node_risk, risk_prediction.max_link_risk)
        if risk_score < min(self.node_risk_threshold, self.link_risk_threshold):
            scope = "none"
            migrate = False
        elif len(target_v_nodes) == 0 and len(target_v_links) > 0:
            scope = "link-only"
            migrate = True
        elif risk_score >= self.full_migration_threshold or risky_nodes_ratio >= self.partial_ratio_threshold:
            scope = "full"
            target_v_nodes = list(v_net.nodes)
            target_v_links = list(v_net.links)
            migrate = True
        else:
            scope = "partial"
            migrate = True

        if not migrate:
            delay = 0
        elif risk_score >= self.full_migration_threshold:
            delay = 0
        else:
            delay = min(self.max_trigger_delay, max(0, int((1.0 - risk_score) * self.max_trigger_delay)))

        estimated_cost = self._estimate_migration_cost(current_solution, target_v_nodes, target_v_links)
        return MigrationPlan(
            migrate=migrate,
            trigger_delay=delay,
            scope=scope,
            target_v_nodes=target_v_nodes,
            target_v_links=target_v_links,
            estimated_cost=estimated_cost,
            risk_score=risk_score,
        )
