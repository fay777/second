from dataclasses import dataclass
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np

from .executor import ElasticReconfigurationExecutor, ExecutionOutcome
from .metrics import MetricsTracker, ReconfigurationRecord
from .mdp.execution_env import ExecutionEnv, ExecutionObservation
from .mdp.planning_env import PlanningEnv, PlanningObservation
from .planner import MigrationPlan, MigrationPlanner
from .risk_predictor import STRiskPredictor
from .topology import RunningService, TimeVaryingTopology, edge_key, graph_copy


TopologyBuilder = Callable[[int], nx.Graph]


@dataclass
class ReconfigurationResult:
    service: RunningService
    plan: MigrationPlan
    post_risk: float
    realized_cost: float
    sla_violated: bool


@dataclass
class MDPTransition:
    observation: Optional[object]
    reward: float
    terminated: bool
    stage: str
    info: Dict[str, object]


class DynamicTopologyEnv:
    def __init__(
        self,
        topology: TimeVaryingTopology,
        service: RunningService,
        topology_builder: TopologyBuilder,
        predictor: STRiskPredictor,
        planner: MigrationPlanner,
        executor: ElasticReconfigurationExecutor,
        planning_env: Optional[PlanningEnv] = None,
        execution_env: Optional[ExecutionEnv] = None,
    ):
        self.topology = topology
        self.service = service
        self.topology_builder = topology_builder
        self.predictor = predictor
        self.planner = planner
        self.executor = executor
        self.planning_env = planning_env or PlanningEnv(
            planner.config.max_trigger_delay,
            planner.config.node_risk_threshold,
            planner.config.link_risk_threshold,
        )
        self.execution_env = execution_env or ExecutionEnv()
        self._initial_history = [graph_copy(graph) for graph in topology.history]
        self._initial_service = service.clone()
        self.time_step = len(topology.history) - 1
        self.metrics = MetricsTracker()
        self.pending_plan: Optional[MigrationPlan] = None
        self.pending_plan_due_time: Optional[int] = None
        self._training_stage = "planning"
        self._training_prediction = None
        self._training_plan: Optional[MigrationPlan] = None
        self._training_service_before: Optional[RunningService] = None
        self._last_execution_outcome: Optional[ExecutionOutcome] = None
        self._waiting_penalty = 0.0

    def reset(self) -> PlanningObservation:
        """Reset one service episode and return the first planning observation."""
        self.topology = TimeVaryingTopology(history=[graph_copy(graph) for graph in self._initial_history])
        self.service = self._initial_service.clone()
        self.time_step = len(self.topology.history) - 1
        self.metrics = MetricsTracker()
        self.pending_plan = None
        self.pending_plan_due_time = None
        self._training_stage = "planning"
        self._training_plan = None
        self._training_service_before = None
        self._last_execution_outcome = None
        self._waiting_penalty = 0.0
        self._training_prediction = self.predictor.predict(self.topology)
        return self._planning_observation(self._training_prediction)

    @property
    def training_stage(self) -> str:
        return self._training_stage

    def _planning_observation(self, prediction) -> PlanningObservation:
        graph = self.topology.current
        total_cpu = sum(float(attrs.get("max_cpu", 1.0)) for _, attrs in graph.nodes(data=True))
        used_cpu = sum(float(attrs.get("cpu", 0.0)) for _, attrs in self.service.virtual_graph.nodes(data=True))
        total_bandwidth = sum(float(attrs.get("max_bandwidth", 1.0)) for _, _, attrs in graph.edges(data=True))
        used_bandwidth = sum(float(attrs.get("bandwidth", 0.0)) for _, _, attrs in self.service.virtual_graph.edges(data=True))
        current_delay = self._service_delay(self.service.deployment)
        expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
        deployed_node_risk = [expected_node_risk.get(node_id, 0.0) for node_id in self.service.deployment.node_mapping.values()]
        deployed_link_risk = [
            expected_link_risk.get(edge_key(*link_id), 0.0)
            for path in self.service.deployment.link_mapping.values()
            for link_id in path
        ]
        future_service_risk = [
            self._deployment_risk_at_horizon(self.service.deployment, node_risk, link_risk)
            for node_risk, link_risk in zip(prediction.future_node_risk, prediction.future_link_risk)
        ]
        service_features = np.array(
            [
                used_cpu / max(total_cpu, 1e-6),
                used_bandwidth / max(total_bandwidth, 1e-6),
                current_delay / max(self.service.max_delay, 1e-6),
                self.service.remaining_lifetime / max(1, self._initial_service.remaining_lifetime),
                self.service.migration_count / 10.0,
            ],
            dtype=np.float32,
        )
        risk_features = np.array(
            [
                np.mean(deployed_node_risk) if deployed_node_risk else 0.0,
                np.mean(deployed_link_risk) if deployed_link_risk else 0.0,
                max(deployed_node_risk, default=0.0),
                max(deployed_link_risk, default=0.0),
                *future_service_risk,
            ],
            dtype=np.float32,
        )
        cost_features = np.array(
            [
                self.service.priority,
                max(0.0, 1.0 - current_delay / max(self.service.max_delay, 1e-6)),
                self.service.disruption_time / 10.0,
            ],
            dtype=np.float32,
        )
        return self.planning_env.build_state(
            service_features,
            risk_features,
            cost_features,
            self.service,
            prediction,
        )

    def _finish_training_decision(
        self,
        plan: MigrationPlan,
        service_before: RunningService,
        prediction,
        migrated: bool,
        failure: bool = False,
    ) -> MDPTransition:
        current_delay = self._service_delay(self.service.deployment)
        sla_violated = failure or current_delay > self.service.max_delay
        if sla_violated:
            self.service.sla_violations += 1
        outcome = self._record_outcome(plan, service_before, prediction, migrated, sla_violated)
        post_risk = outcome.post_risk
        realized_cost = outcome.total_cost
        disruption = self.service.disruption_time - service_before.disruption_time
        future_risk = self._service_future_risk(self.service.deployment, prediction)
        waiting_penalty = self._waiting_penalty
        reward = 1.0 - future_risk - 0.6 * realized_cost - 0.4 * disruption - waiting_penalty
        if sla_violated:
            reward -= 1.5
        if failure:
            reward -= 1.0
        self.service.remaining_lifetime = max(0, self.service.remaining_lifetime - 1)
        if self.service.remaining_lifetime == 0:
            self.service.status = "completed"
            return MDPTransition(None, reward, True, "terminated", {"plan": plan, "failure": failure, "waiting_penalty": waiting_penalty})
        self.step_time()
        self._training_prediction = self.predictor.predict(self.topology)
        self._training_stage = "planning"
        return MDPTransition(
            self._planning_observation(self._training_prediction),
            reward,
            False,
            "planning",
            {"plan": plan, "migrated": migrated, "post_risk": post_risk, "sla_violated": sla_violated, "waiting_penalty": waiting_penalty},
        )

    def planning_step(self, action: int) -> MDPTransition:
        """Apply one planning action; migration actions enter the execution stage."""
        if self._training_stage != "planning":
            raise RuntimeError("Call execution_step until the active reconfiguration is complete.")
        self._last_execution_outcome = None
        self._waiting_penalty = 0.0
        prediction = self._training_prediction or self.predictor.predict(self.topology)
        plan = self.planning_env.plan_from_action(action, self.service, prediction)
        service_before = self.service.clone()
        if plan.migrate and plan.trigger_delay:
            for _ in range(plan.trigger_delay):
                self.service.remaining_lifetime = max(0, self.service.remaining_lifetime - 1)
                if self.service.remaining_lifetime == 0:
                    self.service.status = "completed"
                    return MDPTransition(None, -2.0, True, "terminated", {"plan": plan, "failure": True})
                self.step_time()
                if self._service_delay(self.service.deployment) > self.service.max_delay:
                    self.service.sla_violations += 1
                    self._waiting_penalty += 1.5
                if self._service_unavailable(self.service.deployment):
                    self._waiting_penalty += 1.0
            prediction = self.predictor.predict(self.topology)
            self._training_prediction = prediction
        if not plan.migrate:
            return self._finish_training_decision(plan, service_before, prediction, False)
        if plan.scope == "link-only":
            try:
                migrated = self._execute_plan(plan, prediction)
                return self._finish_training_decision(plan, service_before, prediction, migrated)
            except RuntimeError:
                return self._finish_training_decision(plan, service_before, prediction, False, failure=True)

        try:
            expected_node_risk, _ = prediction.expected_risk_maps()
            observation = self.execution_env.reset(
                self.topology.current,
                self.service,
                plan,
                expected_node_risk,
            )
        except RuntimeError:
            return self._finish_training_decision(plan, service_before, prediction, False, failure=True)
        self._training_stage = "execution"
        self._training_plan = plan
        self._training_service_before = service_before
        if observation is None:
            return self._finish_training_decision(plan, service_before, prediction, False, failure=True)
        return MDPTransition(observation, 0.0, False, "execution", {"plan": plan})

    def execution_step(self, action: int) -> MDPTransition:
        """Place one virtual node selected by the execution policy."""
        if self._training_stage != "execution" or self._training_plan is None or self._training_service_before is None:
            raise RuntimeError("No active execution episode.")
        try:
            observation, done, info = self.execution_env.step(action)
            if not done:
                return MDPTransition(observation, 0.02, False, "execution", info)
            assert self.execution_env.graph is not None
            prediction = self._training_prediction
            expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
            deployment = self.executor.route_links(
                self.execution_env.graph,
                self.service,
                self.execution_env.node_mapping,
                expected_link_risk,
                f"Learned proactive reconfiguration ({self._training_plan.scope})",
                set(self._training_plan.target_v_links),
            )
            outcome = self.executor.assess_deployment(
                self.service,
                deployment,
                expected_node_risk,
                expected_link_risk,
            )
            self._last_execution_outcome = outcome
            self.service.deployment = outcome.deployment
            if outcome.migrated:
                self.service.migration_count += 1
                self.service.migrated_virtual_nodes += outcome.node_migrations
                self.service.rerouted_virtual_links += outcome.link_reroutes
                self.service.disruption_time += 1.0
            return self._finish_training_decision(self._training_plan, self._training_service_before, prediction, outcome.migrated)
        except (RuntimeError, ValueError):
            return self._finish_training_decision(
                self._training_plan,
                self._training_service_before,
                self._training_prediction,
                False,
                failure=True,
            )

    def step_time(self) -> nx.Graph:
        self.time_step += 1
        snapshot = self.topology_builder(self.time_step)
        self.topology.history.append(snapshot)
        return snapshot

    def _service_delay(self, deployment) -> float:
        delay = 0.0
        current_graph = self.topology.current
        for path in deployment.link_mapping.values():
            for link_id in path:
                u, v = edge_key(*link_id)
                if current_graph.has_edge(u, v):
                    delay += float(current_graph.edges[(u, v)].get("delay", 0.0))
        return delay

    def _service_unavailable(self, deployment) -> bool:
        graph = self.topology.current
        for physical_node in deployment.node_mapping.values():
            if not graph.has_node(physical_node) or float(graph.nodes[physical_node].get("fault", 0.0)) >= 1.0:
                return True
        for path in deployment.link_mapping.values():
            for link_id in path:
                u, v = edge_key(*link_id)
                if not graph.has_edge(u, v):
                    return True
                attrs = graph.edges[(u, v)]
                if float(attrs.get("fault", 0.0)) >= 1.0 or float(attrs.get("bandwidth", 0.0)) <= 0.0:
                    return True
        return False

    def _service_post_risk(self, deployment, prediction) -> float:
        expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
        return self._deployment_risk_at_horizon(deployment, expected_node_risk, expected_link_risk)

    def _deployment_risk_at_horizon(self, deployment, node_risk, link_risk) -> float:
        node_scores = [
            node_risk.get(p_node_id, 0.0)
            for p_node_id in deployment.node_mapping.values()
        ]
        link_scores = []
        for path in deployment.link_mapping.values():
            for link_id in path:
                link_scores.append(link_risk.get(edge_key(*link_id), 0.0))
        mean_node = sum(node_scores) / max(1, len(node_scores))
        mean_link = sum(link_scores) / max(1, len(link_scores))
        return max(mean_node, mean_link)

    def _service_future_risk(self, deployment, prediction) -> float:
        expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
        return self._deployment_risk_at_horizon(deployment, expected_node_risk, expected_link_risk)

    def _realized_cost(self, service_before: RunningService, service_after: RunningService) -> float:
        old_nodes = service_before.deployment.node_mapping
        new_nodes = service_after.deployment.node_mapping
        old_links = service_before.deployment.link_mapping
        new_links = service_after.deployment.link_mapping

        node_changes = sum(
            1 for v_node_id, p_node_id in old_nodes.items()
            if new_nodes.get(v_node_id) != p_node_id
        )
        link_changes = sum(
            1 for v_link_id, path in old_links.items()
            if new_links.get(v_link_id) != path
        )
        return min(
            1.0,
            0.6 * node_changes / max(1, len(old_nodes))
            + 0.4 * link_changes / max(1, len(old_links)),
        )

    def _record_outcome(
        self,
        plan: MigrationPlan,
        service_before: RunningService,
        prediction,
        migrated: bool,
        sla_violated: bool,
    ) -> ExecutionOutcome:
        outcome = self._last_execution_outcome
        if outcome is None:
            expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
            outcome = self.executor.assess_deployment(
                service_before,
                self.service.deployment,
                expected_node_risk,
                expected_link_risk,
            )
        self.metrics.add(
            ReconfigurationRecord(
                time_step=self.time_step,
                service_id=self.service.service_id,
                risk_score=plan.risk_score,
                migrated=migrated,
                scope=plan.scope,
                estimated_cost=plan.estimated_cost,
                realized_cost=outcome.total_cost,
                disruption_time=self.service.disruption_time - service_before.disruption_time,
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
        return outcome

    def _execute_plan(self, plan: MigrationPlan, prediction) -> bool:
        try:
            expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
            outcome = self.executor.execute_with_outcome(
                self.topology,
                self.service,
                plan,
                expected_node_risk,
                expected_link_risk,
            )
        except RuntimeError:
            expected_node_risk, expected_link_risk = prediction.expected_risk_maps()
            current_risk = self.executor.deployment_risk(
                self.service.deployment,
                expected_node_risk,
                expected_link_risk,
            )
            outcome = ExecutionOutcome(
                deployment=self.service.deployment.clone(),
                accepted=False,
                reason="infeasible",
                pre_risk=current_risk,
                post_risk=current_risk,
                node_migrations=0,
                link_reroutes=0,
                node_migration_cost=0.0,
                link_reroute_cost=0.0,
            )
        self._last_execution_outcome = outcome
        self.service.deployment = outcome.deployment
        if outcome.migrated:
            self.service.migration_count += 1
            self.service.migrated_virtual_nodes += outcome.node_migrations
            self.service.rerouted_virtual_links += outcome.link_reroutes
            self.service.disruption_time += 0.5 if plan.scope == "link-only" else 1.0
        return outcome.migrated

    def process_current_service(self) -> ReconfigurationResult:
        service_before = self.service.clone()
        self._last_execution_outcome = None
        prediction = self.predictor.predict(self.topology)
        executed_now = False

        if self.pending_plan is not None and self.pending_plan_due_time is not None and self.time_step >= self.pending_plan_due_time:
            plan = self.pending_plan
            self.pending_plan = None
            self.pending_plan_due_time = None
            executed_now = self._execute_plan(plan, prediction)
        else:
            plan = self.planner.plan(self.service, prediction)
            if plan.migrate and plan.trigger_delay == 0:
                executed_now = self._execute_plan(plan, prediction)
            elif plan.migrate and plan.trigger_delay > 0:
                self.pending_plan = plan
                self.pending_plan_due_time = self.time_step + plan.trigger_delay

        if self.pending_plan is not None and not executed_now:
            plan = MigrationPlan(
                migrate=plan.migrate,
                trigger_delay=max(0, self.pending_plan_due_time - self.time_step) if self.pending_plan_due_time is not None else plan.trigger_delay,
                scope=plan.scope,
                target_v_nodes=list(plan.target_v_nodes),
                target_v_links=list(plan.target_v_links),
                estimated_cost=plan.estimated_cost,
                risk_score=plan.risk_score,
            )

        current_delay = self._service_delay(self.service.deployment)
        sla_violated = current_delay > self.service.max_delay
        if sla_violated:
            self.service.sla_violations += 1

        outcome = self._record_outcome(plan, service_before, prediction, executed_now, sla_violated)
        post_risk = outcome.post_risk
        realized_cost = outcome.total_cost
        self.service.remaining_lifetime = max(0, self.service.remaining_lifetime - 1)
        if self.service.remaining_lifetime == 0:
            self.service.status = "completed"

        return ReconfigurationResult(
            service=self.service.clone(),
            plan=plan,
            post_risk=post_risk,
            realized_cost=realized_cost,
            sla_violated=sla_violated,
        )

    def run(self, num_steps: int) -> Dict[str, object]:
        step_results = []
        for _ in range(num_steps):
            if self.service.status != "running":
                break
            self.step_time()
            step_results.append(self.process_current_service())
        return {
            "service": self.service.clone(),
            "metrics": self.metrics.summary(),
            "steps": step_results,
        }
