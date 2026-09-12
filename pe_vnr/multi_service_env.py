"""Shared-resource dynamic SAGIN environment for multiple running VNRs.

The existing ``DynamicTopologyEnv`` remains the single-service HAC worker. This
environment owns arrivals, departures, and the shared CPU/bandwidth ledger used
by paper-scale experiments and by a future service-selection scheduler.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional

import networkx as nx

from .executor import ElasticReconfigurationExecutor
from .metrics import MetricsTracker, ReconfigurationRecord
from .planner import MigrationPlan, MigrationPlanner
from .risk_predictor import STRiskPredictor
from .scenarios import DynamicSAGINScenario
from .service_selector import AllRiskySelector, RiskAwareServiceSelector, ServiceSelectionRecord, TopRiskKSelector
from .topology import RunningService, TimeVaryingTopology, edge_key
from .workload import WorkloadEvent, generate_workload_trace


@dataclass
class MultiServiceConfig:
    arrival_probability: float = 0.45
    max_active_services: int = 12
    min_virtual_nodes: int = 2
    max_virtual_nodes: int = 10
    auto_reconfigure: bool = True
    policy: str = "proactive_heuristic"

    def validate(self) -> None:
        if not 0.0 <= self.arrival_probability <= 1.0:
            raise ValueError("arrival_probability must be in [0, 1]")
        if self.max_active_services < 1:
            raise ValueError("max_active_services must be positive")
        if not 2 <= self.min_virtual_nodes <= self.max_virtual_nodes <= 10:
            raise ValueError("virtual-node bounds must satisfy 2 <= min <= max <= 10")
        if self.policy not in {
            "static", "reactive_full", "proactive_heuristic", "heuristic_all_risky",
            "stgcn_all_risky", "stgcn_topk_heuristic",
        }:
            raise ValueError("unsupported multi-service policy")


@dataclass
class MultiServiceStep:
    time_step: int
    arrivals: int
    admissions: int
    departures: int
    active_services: int
    reconfigurations: int
    sla_violations: int
    unavailable_services: int
    cpu_utilization: float
    bandwidth_utilization: float
    cpu_capacity_degradation: float
    bandwidth_capacity_degradation: float
    prediction_runtime_seconds: float
    selector_candidate_services: int
    selector_selected_services: int
    selector_pending_services: int
    active_pending_plans: int
    runtime_seconds: float


class MultiServiceDynamicEnv:
    def __init__(
        self,
        scenario: DynamicSAGINScenario,
        predictor: STRiskPredictor,
        planner: MigrationPlanner,
        executor: ElasticReconfigurationExecutor,
        config: Optional[MultiServiceConfig] = None,
        history_window: int = 4,
        workload: Optional[List[WorkloadEvent]] = None,
        service_selector: Optional[RiskAwareServiceSelector] = None,
    ):
        self.scenario = scenario
        self.predictor = predictor
        self.planner = planner
        self.executor = executor
        self.config = config or MultiServiceConfig()
        self.config.validate()
        self.history_window = history_window
        self.time_step = history_window - 1
        self.physical_topology = TimeVaryingTopology([])
        self.residual_topology = TimeVaryingTopology([])
        self.services: Dict[str, RunningService] = {}
        self.metrics = MetricsTracker()
        self.arrivals = 0
        self.admissions = 0
        self.departures = 0
        self.rejected_admissions = 0
        self.service_slots = 0
        self.available_service_slots = 0
        self.sla_violation_slots = 0
        self.unavailable_slots = 0
        self.delay_sum = 0.0
        self.step_records: List[MultiServiceStep] = []
        self.pending_plans: Dict[str, tuple] = {}
        self.workload_events = list(workload or [])
        self._workload_provided = workload is not None
        self._events_by_time: Dict[int, List[WorkloadEvent]] = {}
        self._index_workload_events()
        self._raw_current: Optional[nx.Graph] = None
        self.selection_records: List[ServiceSelectionRecord] = []
        if service_selector is not None:
            self.service_selector = service_selector
        elif self.config.policy == "stgcn_topk_heuristic":
            self.service_selector = TopRiskKSelector()
        else:
            self.service_selector = AllRiskySelector()
        self._refresh_graph(rebuild_history=True)

    @property
    def active_services(self) -> List[RunningService]:
        return [service for service in self.services.values() if service.status == "running"]

    def _refresh_graph(self, rebuild_history: bool = False) -> nx.Graph:
        raw = self.scenario.snapshot(self.time_step)
        self._raw_current = nx.Graph(raw)
        graph = nx.Graph(raw)
        for service in self.active_services:
            self._reserve_service(graph, service)
        if rebuild_history:
            physical_history = [nx.Graph(self.scenario.snapshot(step)) for step in range(self.history_window)]
            self.physical_topology = TimeVaryingTopology(physical_history)
        else:
            self.physical_topology.history.append(nx.Graph(raw))
        self.residual_topology = TimeVaryingTopology([graph])
        return graph

    def _index_workload_events(self) -> None:
        self._events_by_time = {}
        for event in self.workload_events:
            self._events_by_time.setdefault(event.time_step, []).append(event)

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

    def _record_reconfiguration(self, service: RunningService, plan: MigrationPlan, outcome, sla_violated: bool) -> None:
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
                sla_violated=sla_violated,
            )
        )

    def _reconfigure(self, service: RunningService, graph: nx.Graph, prediction, plan: MigrationPlan, sla_violated: bool) -> bool:
        if prediction is None:
            expected_node_risk = {node_id: 0.0 for node_id in graph.nodes}
            expected_link_risk = {edge_key(u, v): 0.0 for u, v in graph.edges}
        else:
            expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
        self._release_service(graph, service)
        try:
            # Reactive recovery must restore feasibility even without a future-risk score.
            original_gate = self.executor.config.enforce_risk_reduction
            if prediction is None:
                self.executor.config.enforce_risk_reduction = False
            try:
                outcome = self.executor.execute_with_outcome(
                    TimeVaryingTopology([graph]), service, plan, expected_node_risk, expected_link_risk
                )
            finally:
                self.executor.config.enforce_risk_reduction = original_gate
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
        self._record_reconfiguration(service, plan, outcome, sla_violated)
        return outcome.migrated

    @staticmethod
    def _service_health(service: RunningService, graph: nx.Graph) -> Dict[str, object]:
        delay = 0.0
        unavailable = False
        for physical_node in service.deployment.node_mapping.values():
            if not graph.has_node(physical_node) or float(graph.nodes[physical_node].get("fault", 0.0)) >= 1.0:
                unavailable = True
        for path in service.deployment.link_mapping.values():
            for link_id in path:
                if not graph.has_edge(*link_id):
                    unavailable = True
                    continue
                attrs = graph.edges[link_id]
                delay += float(attrs.get("delay", 0.0))
                if (
                    float(attrs.get("fault", 0.0)) >= 1.0
                    or float(attrs.get("visible_time", 1.0)) <= 0.0
                    or float(attrs.get("max_bandwidth", 0.0)) <= 0.0
                    # A full link is valid for admitted services; only capacity
                    # shrinkage beyond its reservations makes it infeasible.
                    or float(attrs.get("bandwidth", 0.0)) < -1e-6
                ):
                    unavailable = True
        return {"delay": delay, "unavailable": unavailable, "sla_violated": unavailable or delay > service.max_delay}

    def _resource_utilization(self) -> Dict[str, float]:
        assert self._raw_current is not None
        allocated_cpu = sum(float(service.virtual_graph.nodes[v_node].get("cpu", 0.0)) for service in self.active_services for v_node in service.deployment.node_mapping)
        # A virtual-link demand consumes capacity on every physical hop of its path.
        allocated_bandwidth = sum(
            float(service.virtual_graph.edges[v_link].get("bandwidth", 0.0)) * len(path)
            for service in self.active_services
            for v_link, path in service.deployment.link_mapping.items()
        )
        available_cpu = sum(max(0.0, float(attrs.get("cpu", 0.0))) for _, attrs in self._raw_current.nodes(data=True))
        available_bandwidth = sum(max(0.0, float(attrs.get("bandwidth", 0.0))) for _, _, attrs in self._raw_current.edges(data=True))
        nominal_cpu = sum(float(attrs.get("max_cpu", 0.0)) for _, attrs in self._raw_current.nodes(data=True))
        nominal_bandwidth = sum(float(attrs.get("max_bandwidth", 0.0)) for _, _, attrs in self._raw_current.edges(data=True))
        return {
            "cpu": allocated_cpu / max(available_cpu, 1e-6),
            "bandwidth": allocated_bandwidth / max(available_bandwidth, 1e-6),
            "cpu_capacity_degradation": max(0.0, 1.0 - available_cpu / max(nominal_cpu, 1e-6)),
            "bandwidth_capacity_degradation": max(0.0, 1.0 - available_bandwidth / max(nominal_bandwidth, 1e-6)),
        }

    def _plan_for_policy(self, service: RunningService, prediction, health: Dict[str, object]) -> Optional[MigrationPlan]:
        if not self.config.auto_reconfigure or self.config.policy == "static":
            return None
        if self.config.policy == "reactive_full":
            if not health["sla_violated"]:
                return None
            self.pending_plans.pop(service.service_id, None)
            return MigrationPlan(True, 0, "full", list(service.virtual_graph.nodes), list(service.virtual_graph.edges), 1.0, 0.0)
        if prediction is None:
            return None
        pending = self.pending_plans.get(service.service_id)
        if pending is not None:
            plan, due_time = pending
            if self.time_step >= due_time:
                self.pending_plans.pop(service.service_id, None)
                return plan
            return None
        observed = self.planner.plan(service, prediction)
        if not observed.migrate:
            return None
        if observed.trigger_delay > 0:
            self.pending_plans[service.service_id] = (observed, self.time_step + observed.trigger_delay)
            return None
        return observed

    def _uses_prediction(self) -> bool:
        return self.config.auto_reconfigure and self.config.policy in {
            "proactive_heuristic",
            "heuristic_all_risky",
            "stgcn_all_risky",
            "stgcn_topk_heuristic",
        }

    def step(self) -> MultiServiceStep:
        started = perf_counter()
        self.time_step += 1
        graph = self._refresh_graph()
        arrivals = admissions = reconfigurations = 0
        for event in self._events_by_time.get(self.time_step, []):
            arrivals += 1
            self.arrivals += 1
            if len(self.active_services) < self.config.max_active_services and self._admit(event.service.clone(), graph):
                admissions += 1
            else:
                self.rejected_admissions += 1
        health_records = {service.service_id: self._service_health(service, graph) for service in self.active_services}
        sla_violations = sum(int(record["sla_violated"]) for record in health_records.values())
        unavailable_services = sum(int(record["unavailable"]) for record in health_records.values())
        self.service_slots += len(health_records)
        self.sla_violation_slots += sla_violations
        self.unavailable_slots += unavailable_services
        self.available_service_slots += len(health_records) - unavailable_services
        self.delay_sum += sum(float(record["delay"]) for record in health_records.values())
        prediction = None
        prediction_runtime = 0.0
        selection_records: List[ServiceSelectionRecord] = []
        selected_service_ids: List[str] = []
        selector_selected_count = 0
        due_service_ids: List[str] = []
        if self._uses_prediction():
            prediction_started = perf_counter()
            prediction = self.predictor.predict(self.physical_topology)
            prediction_runtime = perf_counter() - prediction_started
            if self.config.policy == "proactive_heuristic":
                # Preserve the original baseline: every active service reaches its planner.
                selected_service_ids = [service.service_id for service in self.active_services]
            else:
                selection = self.service_selector.select(self.active_services, prediction, graph, self.pending_plans)
                selection_records = selection.records
                selected_service_ids = selection.selected_service_ids
                selector_selected_count = len(selected_service_ids)
            due_service_ids = [
                service_id
                for service_id, (_, due_time) in self.pending_plans.items()
                if due_time <= self.time_step and service_id in self.services
            ]
            self.selection_records.extend(selection_records)
        for service in self.active_services:
            health = health_records[service.service_id]
            if health["sla_violated"]:
                service.sla_violations += 1
        if self.config.policy == "reactive_full":
            service_order = [service.service_id for service in self.active_services]
        else:
            due_service_set = set(due_service_ids)
            service_order = due_service_ids + [
                service_id for service_id in selected_service_ids if service_id not in due_service_set
            ]
        for service_id in service_order:
            service = self.services.get(service_id)
            if service is None or service.status != "running":
                continue
            health = health_records[service.service_id]
            plan = self._plan_for_policy(service, prediction, health)
            if plan is not None:
                reconfigurations += int(self._reconfigure(service, graph, prediction, plan, bool(health["sla_violated"])))
        departures = 0
        for service_id, service in list(self.services.items()):
            service.remaining_lifetime -= 1
            if service.remaining_lifetime <= 0:
                service.status = "completed"
                del self.services[service_id]
                self.pending_plans.pop(service_id, None)
                departures += 1
                self.departures += 1
        # Rebuild the current snapshot after admissions, reconfigurations, and releases.
        residual_graph = nx.Graph(self.scenario.snapshot(self.time_step))
        for service in self.active_services:
            self._reserve_service(residual_graph, service)
        self.residual_topology = TimeVaryingTopology([residual_graph])
        utilization = self._resource_utilization()
        record = MultiServiceStep(
            self.time_step, arrivals, admissions, departures, len(self.active_services), reconfigurations,
            sla_violations, unavailable_services, utilization["cpu"], utilization["bandwidth"],
            utilization["cpu_capacity_degradation"], utilization["bandwidth_capacity_degradation"],
            prediction_runtime, sum(record.rank is not None for record in selection_records),
            selector_selected_count, sum(record.has_pending_plan for record in selection_records),
            len(self.pending_plans), perf_counter() - started,
        )
        self.step_records.append(record)
        return record

    def run(self, steps: int) -> Dict[str, object]:
        if not self._workload_provided:
            self.workload_events = generate_workload_trace(self.scenario, self.config, self.time_step, steps)
            self._index_workload_events()
        records = [self.step() for _ in range(steps)]
        total_step_runtime = sum(record.runtime_seconds for record in records)
        return {
            "steps": records,
            "services": {service_id: service.clone() for service_id, service in self.services.items()},
            "selection_records": list(self.selection_records),
            "metrics": self.metrics.summary(),
            "arrivals": self.arrivals,
            "admissions": self.admissions,
            "rejected_admissions": self.rejected_admissions,
            "admission_rate": self.admissions / max(1, self.arrivals),
            "departures": self.departures,
            "active_services": len(self.active_services),
            "service_slots": self.service_slots,
            "sla_violations": self.sla_violation_slots,
            "sla_violation_rate": self.sla_violation_slots / max(1, self.service_slots),
            "availability": self.available_service_slots / max(1, self.service_slots),
            "avg_end_to_end_delay": self.delay_sum / max(1, self.service_slots),
            "avg_cpu_utilization": sum(record.cpu_utilization for record in records) / max(1, len(records)),
            "avg_bandwidth_utilization": sum(record.bandwidth_utilization for record in records) / max(1, len(records)),
            "avg_cpu_capacity_degradation": sum(record.cpu_capacity_degradation for record in records) / max(1, len(records)),
            "avg_bandwidth_capacity_degradation": sum(record.bandwidth_capacity_degradation for record in records) / max(1, len(records)),
            "prediction_runtime_seconds": sum(record.prediction_runtime_seconds for record in records),
            "selector_candidate_count": sum(record.selector_candidate_services for record in records),
            "selector_selected_count": sum(record.selector_selected_services for record in records),
            "selector_pending_count": sum(record.selector_pending_services for record in records),
            "avg_active_pending_plans": sum(record.active_pending_plans for record in records) / max(1, len(records)),
            "max_active_pending_plans": max((record.active_pending_plans for record in records), default=0),
            "runtime_seconds": total_step_runtime,
            "avg_step_runtime_seconds": total_step_runtime / max(1, len(records)),
        }
