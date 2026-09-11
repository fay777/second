from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx

from .config import ExecutionConfig
from .planner import MigrationPlan
from .topology import Deployment, RunningService, TimeVaryingTopology, edge_key


@dataclass
class ExecutionOutcome:
    deployment: Deployment
    accepted: bool
    reason: str
    pre_risk: float
    post_risk: float
    node_migrations: int
    link_reroutes: int
    node_migration_cost: float
    link_reroute_cost: float

    @property
    def risk_reduction(self) -> float:
        return self.pre_risk - self.post_risk

    @property
    def migrated(self) -> bool:
        return self.accepted and (self.node_migrations > 0 or self.link_reroutes > 0)

    @property
    def total_cost(self) -> float:
        return min(1.0, self.node_migration_cost + self.link_reroute_cost)


class ElasticReconfigurationExecutor:
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def _candidate_nodes(self, graph: nx.Graph, cpu_demand: float, node_risk: Dict[int, float], blocked: Set[int]) -> List[int]:
        candidates = []
        for node_id, attrs in graph.nodes(data=True):
            if node_id in blocked:
                continue
            if (
                float(attrs.get("cpu", 0.0)) >= cpu_demand
                and float(attrs.get("fault", 0.0)) < 1.0
                and float(attrs.get("available", 1.0)) > 0.0
                and float(attrs.get("energy", 1.0)) > 0.05
            ):
                candidates.append(node_id)
        return sorted(
            candidates,
            key=lambda node_id: (
                node_risk.get(node_id, 0.0),
                -float(graph.nodes[node_id].get("cpu", 0.0)),
                float(graph.nodes[node_id].get("queue", 0.0)),
            ),
        )

    def _reserve_node(self, graph: nx.Graph, node_id: int, cpu_demand: float) -> None:
        graph.nodes[node_id]["cpu"] = float(graph.nodes[node_id].get("cpu", 0.0)) - cpu_demand

    def _find_best_path(self, graph: nx.Graph, source: int, target: int, bandwidth_demand: float, link_risk: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
        """Route once on the feasible subgraph instead of enumerating simple paths."""
        feasible_graph = nx.Graph()
        feasible_graph.add_nodes_from(
            (node_id, attrs)
            for node_id, attrs in graph.nodes(data=True)
            if float(attrs.get("fault", 0.0)) < 1.0
        )
        for u, v, attrs in graph.edges(data=True):
            if u not in feasible_graph or v not in feasible_graph:
                continue
            if (
                float(attrs.get("fault", 0.0)) >= 1.0
                or float(attrs.get("bandwidth", 0.0)) < bandwidth_demand
                or float(attrs.get("visible_time", 1.0)) <= 0.0
            ):
                continue
            link_id = edge_key(u, v)
            weight = (
                self.config.risk_weight * link_risk.get(link_id, 0.0)
                + 0.05 * float(attrs.get("delay", 0.0))
                + 0.02 * float(attrs.get("loss", 0.0))
            )
            feasible_graph.add_edge(u, v, routing_weight=weight)
        try:
            path = nx.shortest_path(feasible_graph, source, target, weight="routing_weight", method="dijkstra")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return [edge_key(u, v) for u, v in zip(path[:-1], path[1:])]

    def _reserve_path(self, graph: nx.Graph, path: List[Tuple[int, int]], bandwidth_demand: float) -> None:
        for link_id in path:
            graph.edges[link_id]["bandwidth"] = float(graph.edges[link_id].get("bandwidth", 0.0)) - bandwidth_demand

    @staticmethod
    def deployment_risk(
        deployment: Deployment,
        node_risk: Dict[int, float],
        link_risk: Dict[Tuple[int, int], float],
    ) -> float:
        node_scores = [node_risk.get(node_id, 0.0) for node_id in deployment.node_mapping.values()]
        link_scores = [
            link_risk.get(edge_key(*link_id), 0.0)
            for path in deployment.link_mapping.values()
            for link_id in path
        ]
        return max(
            sum(node_scores) / max(1, len(node_scores)),
            sum(link_scores) / max(1, len(link_scores)),
        )

    def assess_deployment(
        self,
        service: RunningService,
        candidate: Deployment,
        node_risk: Dict[int, float],
        link_risk: Dict[Tuple[int, int], float],
    ) -> ExecutionOutcome:
        """Accept a candidate only when it changes deployment and improves risk."""
        old = service.deployment
        node_migrations = sum(
            1 for v_node_id, old_node_id in old.node_mapping.items()
            if candidate.node_mapping.get(v_node_id) != old_node_id
        )
        link_reroutes = sum(
            1 for v_link_id, old_path in old.link_mapping.items()
            if candidate.link_mapping.get(v_link_id) != old_path
        )
        pre_risk = self.deployment_risk(old, node_risk, link_risk)
        post_risk = self.deployment_risk(candidate, node_risk, link_risk)
        node_cost = self.config.node_migration_cost_weight * node_migrations / max(1, len(old.node_mapping))
        link_cost = self.config.link_reroute_cost_weight * link_reroutes / max(1, len(old.link_mapping))
        changed = node_migrations > 0 or link_reroutes > 0
        improved = post_risk <= pre_risk - self.config.min_risk_improvement
        accepted = changed and (improved or not self.config.enforce_risk_reduction)
        if not changed:
            reason = "unchanged"
        elif accepted:
            reason = "accepted"
        else:
            reason = "insufficient-risk-reduction"
        return ExecutionOutcome(
            deployment=candidate if accepted else old.clone(),
            accepted=accepted,
            reason=reason,
            pre_risk=pre_risk,
            post_risk=post_risk if accepted else pre_risk,
            node_migrations=node_migrations if accepted else 0,
            link_reroutes=link_reroutes if accepted else 0,
            node_migration_cost=node_cost if accepted else 0.0,
            link_reroute_cost=link_cost if accepted else 0.0,
        )

    def route_links(
        self,
        graph: nx.Graph,
        service: RunningService,
        node_mapping: Dict[int, int],
        link_risk: Dict[Tuple[int, int], float],
        description: str,
        reroute_v_links: Set[Tuple[int, int]] = None,
    ) -> Deployment:
        """Route only affected virtual links and preserve unaffected paths."""
        deployment = Deployment(node_mapping=dict(node_mapping), description=description)
        reroute_v_links = set(service.virtual_graph.edges) if reroute_v_links is None else set(reroute_v_links)
        for u, v, attrs in service.virtual_graph.edges(data=True):
            v_link_id = (u, v)
            if v_link_id not in reroute_v_links and v_link_id in service.deployment.link_mapping:
                path = list(service.deployment.link_mapping[v_link_id])
                demand = float(attrs.get("bandwidth", 0.0))
                for link_id in path:
                    if float(graph.edges[link_id].get("bandwidth", 0.0)) < demand:
                        raise RuntimeError(f"Preserved path for virtual link {v_link_id} is no longer feasible.")
                deployment.link_mapping[v_link_id] = path
                self._reserve_path(graph, path, demand)
                continue
            source = deployment.node_mapping[u]
            target = deployment.node_mapping[v]
            demand = float(attrs.get("bandwidth", 0.0))
            path = self._find_best_path(graph, source, target, demand, link_risk)
            if not path:
                raise RuntimeError(f"No feasible physical path for virtual link {(u, v)}.")
            deployment.link_mapping[(u, v)] = path
            self._reserve_path(graph, path, demand)
        return deployment

    def execute(self, topology: TimeVaryingTopology, service: RunningService, plan: MigrationPlan, node_risk: Dict[int, float], link_risk: Dict[Tuple[int, int], float]) -> Deployment:
        if not plan.migrate:
            deployment = service.deployment.clone()
            deployment.description = "Keep current stable deployment"
            return deployment

        graph = nx.Graph(topology.current)
        old_mapping = service.deployment.node_mapping
        fixed_nodes = {}
        if plan.scope == "link-only":
            fixed_nodes = dict(old_mapping)
        elif plan.scope == "partial":
            fixed_nodes = {v_node_id: p_node_id for v_node_id, p_node_id in old_mapping.items() if v_node_id not in set(plan.target_v_nodes)}

        new_deployment = Deployment(description=f"Proactive reconfiguration ({plan.scope})")
        used_nodes = set()
        for v_node_id, p_node_id in fixed_nodes.items():
            cpu_demand = float(service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
            if float(graph.nodes[p_node_id].get("cpu", 0.0)) < cpu_demand:
                raise RuntimeError(f"Fixed node {p_node_id} is no longer feasible for virtual node {v_node_id}.")
            new_deployment.node_mapping[v_node_id] = p_node_id
            self._reserve_node(graph, p_node_id, cpu_demand)
            used_nodes.add(p_node_id)

        for v_node_id in service.virtual_graph.nodes:
            if v_node_id in new_deployment.node_mapping:
                continue
            cpu_demand = float(service.virtual_graph.nodes[v_node_id].get("cpu", 0.0))
            # A targeted migration must not select its original host again.
            blocked = set(used_nodes)
            if v_node_id in old_mapping:
                blocked.add(old_mapping[v_node_id])
            candidates = self._candidate_nodes(graph, cpu_demand, node_risk, blocked)
            if not candidates:
                raise RuntimeError(f"No feasible physical host for virtual node {v_node_id}.")
            target = candidates[0]
            new_deployment.node_mapping[v_node_id] = target
            self._reserve_node(graph, target, cpu_demand)
            used_nodes.add(target)

        return self.route_links(
            graph,
            service,
            new_deployment.node_mapping,
            link_risk,
            new_deployment.description,
            set(plan.target_v_links),
        )

    def execute_with_outcome(
        self,
        topology: TimeVaryingTopology,
        service: RunningService,
        plan: MigrationPlan,
        node_risk: Dict[int, float],
        link_risk: Dict[Tuple[int, int], float],
    ) -> ExecutionOutcome:
        candidate = self.execute(topology, service, plan, node_risk, link_risk)
        return self.assess_deployment(service, candidate, node_risk, link_risk)
