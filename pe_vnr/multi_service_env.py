"""Shared-resource dynamic SAGIN environment for multiple running VNRs.

The existing ``DynamicTopologyEnv`` remains the single-service HAC worker. This
environment owns arrivals, departures, and the shared CPU/bandwidth ledger used
by paper-scale experiments and by a future service-selection scheduler.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx

from .executor import ElasticReconfigurationExecutor
from .metrics import MetricsTracker, ReconfigurationRecord
from .planner import MigrationPlan, MigrationPlanner
from .risk_predictor import STRiskPredictor
from .scenarios import DynamicSAGINScenario
from .topology import RunningService, TimeVaryingTopology, edge_key


@dataclass
class MultiServiceConfig:
    arrival_probability: float = 0.45
    max_active_services: int = 12
    min_virtual_nodes: int = 2
    max_virtual_nodes: int = 10
    auto_reconfigure: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.arrival_probability <= 1.0:
            raise ValueError("arrival_probability must be in [0, 1]")
        if self.max_active_services < 1:
            raise ValueError("max_active_services must be positive")
        if not 2 <= self.min_virtual_nodes <= self.max_virtual_nodes <= 10:
            raise ValueError("virtual-node bounds must satisfy 2 <= min <= max <= 10")


@dataclass
class MultiServiceStep:
    time_step: int
    arrivals: int
    admissions: int
    departures: int
    active_services: int
    reconfigurations: int


class MultiServiceDynamicEnv:
    def __init__(
        self,
        scenario: DynamicSAGINScenario,
        predictor: STRiskPredictor,
        planner: MigrationPlanner,
        executor: ElasticReconfigurationExecutor,
        config: Optional[MultiServiceConfig] = None,
        history_window: int = 4,
    ):
        self.scenario = scenario
        self.predictor = predictor
        self.planner = planner
        self.executor = executor
        self.config = config or MultiServiceConfig()
        self.config.validate()
        self.history_window = history_window
        self.time_step = history_window - 1
        self.topology = TimeVaryingTopology([])
        self.services: Dict[str, RunningService] = {}
        self.metrics = MetricsTracker()
        self.arrivals = 0
        self.admissions = 0
        self.departures = 0
        self._next_service_index = 1
        self._refresh_graph(rebuild_history=True)

    @property
    def active_services(self) -> List[RunningService]:
        return [service for service in self.services.values() if service.status == "running"]

    def _refresh_graph(self, rebuild_history: bool = False) -> nx.Graph:
        raw = self.scenario.snapshot(self.time_step)
        graph = nx.Graph(raw)
        for service in self.active_services:
            self._reserve_service(graph, service)
        if rebuild_history:
            history = []
            for step in range(self.history_window):
                snapshot = nx.Graph(self.scenario.snapshot(step))
                if step == self.time_step:
                    for service in self.active_services:
                        self._reserve_service(snapshot, service)
                history.append(snapshot)
            self.topology = TimeVaryingTopology(history)
        else:
            self.topology.history.append(graph)
        return graph

    @staticmethod
    def _reserve_service(graph: nx.Graph, service: RunningService) -> None:
        for v_node, physical_node in service.deployment.node_mapping.items():
            if graph.has_node(physical_node):
                demand = float(service.virtual_graph.nodes[v_node].get("cpu", 0.0))
                graph.nodes[physical_node]["cpu"] = float(graph.nodes[physical_node].get("cpu", 0.0)) - demand
        for v_link, path in service.deployment.link_mapping.items():
            demand = float(service.virtual_graph.edges[v_link].get("bandwidth", 0.0))
            for link_id in path:
                if graph.has_edge(*link_id):
                    graph.edges[link_id]["bandwidth"] = float(graph.edges[link_id].get("bandwidth", 0.0)) - demand

    @staticmethod
    def _release_service(graph: nx.Graph, service: RunningService) -> None:
        for v_node, physical_node in service.deployment.node_mapping.items():
            if graph.has_node(physical_node):
                demand = float(service.virtual_graph.nodes[v_node].get("cpu", 0.0))
                graph.nodes[physical_node]["cpu"] = float(graph.nodes[physical_node].get("cpu", 0.0)) + demand
        for v_link, path in service.deployment.link_mapping.items():
            demand = float(service.virtual_graph.edges[v_link].get("bandwidth", 0.0))
            for link_id in path:
                if graph.has_edge(*link_id):
                    graph.edges[link_id]["bandwidth"] = float(graph.edges[link_id].get("bandwidth", 0.0)) + demand

    def _new_service(self) -> RunningService:
        service_id = f"svc-{self._next_service_index:04d}"
        self._next_service_index += 1
        size = self.config.min_virtual_nodes + (
            (self._next_service_index * 7 + self.time_step) % (self.config.max_virtual_nodes - self.config.min_virtual_nodes + 1)
        )
        return self.scenario.build_service(service_id, size, self.time_step)

    def _admit(self, service: RunningService, graph: nx.Graph) -> bool:
        nodes = list(service.virtual_graph.nodes)
        links = list(service.virtual_graph.edges)
        plan = MigrationPlan(True, 0, "full", nodes, links, 1.0, 0.0)
        empty_prediction = ({node: 0.0 for node in graph.nodes}, {edge_key(u, v): 0.0 for u, v in graph.edges})
        try:
            deployment = self.executor.execute(
                TimeVaryingTopology([graph]), service, plan, empty_prediction[0], empty_prediction[1]
            )
        except RuntimeError:
            return False
        service.deployment = deployment
        self.services[service.service_id] = service
        self._reserve_service(graph, service)
        self.admissions += 1
        return True

    def _record_reconfiguration(self, service: RunningService, plan: MigrationPlan, outcome) -> None:
        self.metrics.add(
            ReconfigurationRecord(
                time_step=self.time_step,
                service_id=service.service_id,
                risk_score=plan.risk_score,
                migrated=outcome.migrated,
                scope=plan.scope,
                estimated_cost=plan.estimated_cost,
                realized_cost=outcome.total_cost,
                disruption_time=0.5 if plan.scope == "link-only" and outcome.migrated else float(outcome.migrated),
                post_risk=outcome.post_risk,
                risk_reduction=outcome.risk_reduction,
                node_migrations=outcome.node_migrations,
                link_reroutes=outcome.link_reroutes,
                node_migration_cost=outcome.node_migration_cost,
                link_reroute_cost=outcome.link_reroute_cost,
                accepted=outcome.accepted,
                sla_violated=False,
            )
        )

    def _reconfigure(self, service: RunningService, graph: nx.Graph, prediction) -> bool:
        plan = self.planner.plan(service, prediction)
        if not plan.migrate or plan.trigger_delay:
            return False
        expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
        self._release_service(graph, service)
        try:
            outcome = self.executor.execute_with_outcome(
                TimeVaryingTopology([graph]), service, plan, expected_node_risk, expected_link_risk
            )
        except RuntimeError:
            self._reserve_service(graph, service)
            return False
        if outcome.migrated:
            service.deployment = outcome.deployment
            service.migration_count += 1
            service.migrated_virtual_nodes += outcome.node_migrations
            service.rerouted_virtual_links += outcome.link_reroutes
            service.disruption_time += 0.5 if plan.scope == "link-only" else 1.0
        self._reserve_service(graph, service)
        self._record_reconfiguration(service, plan, outcome)
        return outcome.migrated

    def step(self) -> MultiServiceStep:
        self.time_step += 1
        graph = self._refresh_graph()
        prediction = self.predictor.predict(self.topology)
        arrivals = admissions = reconfigurations = 0
        if len(self.active_services) < self.config.max_active_services:
            deterministic_draw = ((self.time_step * 37 + self._next_service_index * 17) % 100) / 100.0
            if deterministic_draw < self.config.arrival_probability:
                arrivals = 1
                self.arrivals += 1
                service = self._new_service()
                if self._admit(service, graph):
                    admissions = 1
        if self.config.auto_reconfigure:
            for service in list(self.active_services):
                reconfigurations += int(self._reconfigure(service, graph, prediction))
        departures = 0
        for service_id, service in list(self.services.items()):
            service.remaining_lifetime -= 1
            if service.remaining_lifetime <= 0:
                service.status = "completed"
                del self.services[service_id]
                departures += 1
                self.departures += 1
        # Rebuild the current snapshot after admissions, reconfigurations, and releases.
        self.topology.history[-1] = nx.Graph(self.scenario.snapshot(self.time_step))
        for service in self.active_services:
            self._reserve_service(self.topology.history[-1], service)
        return MultiServiceStep(self.time_step, arrivals, admissions, departures, len(self.active_services), reconfigurations)

    def run(self, steps: int) -> Dict[str, object]:
        records = [self.step() for _ in range(steps)]
        return {
            "steps": records,
            "services": {service_id: service.clone() for service_id, service in self.services.items()},
            "metrics": self.metrics.summary(),
            "arrivals": self.arrivals,
            "admissions": self.admissions,
            "departures": self.departures,
            "active_services": len(self.active_services),
        }
