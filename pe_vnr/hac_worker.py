"""Stateless per-service Upper/Lower HAC worker for the multi-VNR environment.

The worker never advances time and never mutates the shared resource ledger.
``MultiServiceDynamicEnv`` owns the release/commit/rollback transaction.
"""

from dataclasses import dataclass, field
from math import ceil
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import torch

from .executor import ElasticReconfigurationExecutor, ExecutionOutcome
from .mdp.actor_critic import ExecutionActorCritic, PlanningActorCritic
from .mdp.execution_env import ExecutionEnv
from .mdp.planning_env import PlanningEnv, PlanningObservation
from .planner import MigrationPlan
from .topology import RiskPrediction, RunningService, TimeVaryingTopology, edge_key


@dataclass(frozen=True)
class HACDecision:
    action: int
    migrate: bool
    delay: int
    scope: str
    plan: Optional[MigrationPlan]
    observation: Optional[PlanningObservation] = None
    decision_id: int = -1


@dataclass(frozen=True)
class PendingHACDecision:
    service_id: str
    due_time: int
    scope: str
    created_at: int
    decision_id: int


@dataclass
class HACExecutionResult:
    plan: Optional[MigrationPlan]
    outcome: ExecutionOutcome
    lower_decisions: int
    lower_success: bool
    routing_success: bool
    lower_trace: List["LowerTransition"] = field(default_factory=list)


@dataclass(frozen=True)
class LowerTransition:
    """One lower-placement decision exposed for external rollout collection."""

    observation: object
    action: int
    virtual_node: int
    physical_node: int
    done: bool


UpperPolicy = Callable[[PlanningObservation], int]
LowerPolicy = Callable[[object], int]


class MultiServiceHACWorkerAdapter:
    """Apply Upper/Lower HAC to one selected service without a global clock."""

    def __init__(
        self,
        executor: ElasticReconfigurationExecutor,
        planning_env: Optional[PlanningEnv] = None,
        upper_model: Optional[PlanningActorCritic] = None,
        lower_model: Optional[ExecutionActorCritic] = None,
        upper_policy: Optional[UpperPolicy] = None,
        lower_policy: Optional[LowerPolicy] = None,
        device: str = "cpu",
        partial_fraction: float = 0.5,
    ):
        if not 0.0 < partial_fraction <= 1.0:
            raise ValueError("partial_fraction must be in (0, 1].")
        self.executor = executor
        self.planning_env = planning_env or PlanningEnv()
        self.upper_model = upper_model
        self.lower_model = lower_model
        self.upper_policy = upper_policy
        self.lower_policy = lower_policy
        self.device = torch.device(device)
        self.partial_fraction = partial_fraction

    @staticmethod
    def _service_delay(service: RunningService, graph: nx.Graph) -> float:
        return sum(
            float(graph.edges[edge_key(*link_id)].get("delay", 0.0))
            for path in service.deployment.link_mapping.values()
            for link_id in path
            if graph.has_edge(*edge_key(*link_id))
        )

    @staticmethod
    def _deployment_risk(service: RunningService, node_risk: Dict[int, float], link_risk: Dict) -> float:
        return ElasticReconfigurationExecutor.deployment_risk(service.deployment, node_risk, link_risk)

    def upper_observation(
        self,
        service: RunningService,
        residual_graph: nx.Graph,
        prediction: RiskPrediction,
        physical_graph: Optional[nx.Graph] = None,
    ) -> PlanningObservation:
        capacity_graph = physical_graph if physical_graph is not None else residual_graph
        total_cpu = sum(float(attrs.get("max_cpu", 1.0)) for _, attrs in capacity_graph.nodes(data=True))
        total_bandwidth = sum(float(attrs.get("max_bandwidth", 1.0)) for _, _, attrs in capacity_graph.edges(data=True))
        available_cpu = sum(max(0.0, float(attrs.get("cpu", 0.0))) for _, attrs in capacity_graph.nodes(data=True))
        available_bandwidth = sum(max(0.0, float(attrs.get("bandwidth", 0.0))) for _, _, attrs in capacity_graph.edges(data=True))
        residual_cpu = sum(max(0.0, float(attrs.get("cpu", 0.0))) for _, attrs in residual_graph.nodes(data=True))
        residual_bandwidth = sum(max(0.0, float(attrs.get("bandwidth", 0.0))) for _, _, attrs in residual_graph.edges(data=True))
        demand_cpu = sum(float(attrs.get("cpu", 0.0)) for _, attrs in service.virtual_graph.nodes(data=True))
        demand_bandwidth = sum(float(attrs.get("bandwidth", 0.0)) for _, _, attrs in service.virtual_graph.edges(data=True))
        expected_node, expected_link = prediction.expected_risk_maps()
        node_scores = [expected_node.get(node_id, 0.0) for node_id in service.deployment.node_mapping.values()]
        link_scores = [
            expected_link.get(edge_key(*link_id), 0.0)
            for path in service.deployment.link_mapping.values()
            for link_id in path
        ]
        horizon_risks = [
            self._deployment_risk(service, node_map, link_map)
            for node_map, link_map in zip(prediction.future_node_risk, prediction.future_link_risk)
        ]
        delay = self._service_delay(service, residual_graph)
        service_features = np.array(
            [
                demand_cpu / max(total_cpu, 1e-6),
                demand_bandwidth / max(total_bandwidth, 1e-6),
                delay / max(service.max_delay, 1e-6),
                min(1.0, service.remaining_lifetime / 100.0),
                min(1.0, service.migration_count / 10.0),
            ],
            dtype=np.float32,
        )
        risk_features = np.array(
            [
                np.mean(node_scores) if node_scores else 0.0,
                np.mean(link_scores) if link_scores else 0.0,
                max(node_scores, default=0.0),
                max(link_scores, default=0.0),
                *horizon_risks,
            ],
            dtype=np.float32,
        )
        cost_features = np.array(
            [
                service.priority,
                max(0.0, 1.0 - delay / max(service.max_delay, 1e-6)),
                min(1.0, service.disruption_time / 10.0),
                max(0.0, 1.0 - residual_cpu / max(available_cpu, 1e-6)),
                max(0.0, 1.0 - residual_bandwidth / max(available_bandwidth, 1e-6)),
            ],
            dtype=np.float32,
        )
        observation = self.planning_env.build_state(service_features, risk_features, cost_features)
        # Upper HAC must decide scope itself. Target localization happens later.
        mask = np.zeros_like(observation.action_mask)
        mask[self.planning_env.encode_action(0, 0, 0)] = True
        for delay_value in range(self.planning_env.max_delay + 1):
            if delay_value >= service.remaining_lifetime:
                continue
            for scope_index in range(len(self.planning_env.scopes)):
                mask[self.planning_env.encode_action(1, delay_value, scope_index)] = True
        return PlanningObservation(state=observation.state, action_mask=mask)

    def _select_upper_action(self, observation: PlanningObservation) -> int:
        if self.upper_policy is not None:
            return int(self.upper_policy(observation))
        if self.upper_model is None:
            return self.planning_env.encode_action(0, 0, 0)
        with torch.no_grad():
            state = torch.as_tensor(observation.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(observation.action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            return int(self.upper_model.act(state).masked_fill(~mask, -1e9).argmax(dim=-1).item())

    def _localize_scope(
        self,
        service: RunningService,
        prediction: RiskPrediction,
        scope: str,
    ) -> MigrationPlan:
        expected_node, expected_link = prediction.expected_risk_maps()
        future_nodes = prediction.future_node_risk or [expected_node]
        future_links = prediction.future_link_risk or [expected_link]
        risk_score = self._deployment_risk(service, expected_node, expected_link)
        node_rank = sorted(
            service.deployment.node_mapping,
            key=lambda v_node: max(risk.get(service.deployment.node_mapping[v_node], 0.0) for risk in future_nodes),
            reverse=True,
        )
        link_rank = sorted(
            service.deployment.link_mapping,
            key=lambda v_link: max(
                (risk.get(edge_key(*link_id), 0.0) for risk in future_links for link_id in service.deployment.link_mapping[v_link]),
                default=0.0,
            ),
            reverse=True,
        )
        if scope == "link-only":
            node_targets: List[int] = []
            link_targets = link_rank[:1] if link_rank else []
        elif scope == "partial":
            node_targets = node_rank[: max(1, ceil(len(node_rank) * self.partial_fraction))]
            target_set = set(node_targets)
            link_targets = [v_link for v_link in service.virtual_graph.edges if set(v_link) & target_set]
        elif scope == "full":
            node_targets = list(service.virtual_graph.nodes)
            link_targets = list(service.virtual_graph.edges)
        else:
            raise ValueError(f"Unsupported HAC scope: {scope}")
        cost = min(
            1.0,
            0.6 * len(node_targets) / max(1, len(service.deployment.node_mapping))
            + 0.4 * len(link_targets) / max(1, len(service.deployment.link_mapping)),
        )
        return MigrationPlan(True, 0, scope, node_targets, link_targets, cost, risk_score)

    def upper_decide(
        self,
        service: RunningService,
        physical_prediction: RiskPrediction,
        residual_graph: nx.Graph,
        physical_graph: Optional[nx.Graph] = None,
        action: Optional[int] = None,
        decision_id: int = -1,
    ) -> HACDecision:
        observation = self.upper_observation(service, residual_graph, physical_prediction, physical_graph)
        selected_action = self._select_upper_action(observation) if action is None else action
        if not observation.action_mask[selected_action]:
            raise ValueError("Upper policy selected a masked action.")
        migrate, delay, scope_index = self.planning_env.decode_action(selected_action)
        if migrate == 0:
            return HACDecision(selected_action, False, 0, "none", None, observation, decision_id)
        scope = self.planning_env.scopes[scope_index]
        # Targets are intentionally localized only at the actual execution slot.
        return HACDecision(selected_action, True, delay, scope, None, observation, decision_id)

    def execution_plan(
        self,
        service: RunningService,
        scope: str,
        physical_prediction: RiskPrediction,
    ) -> MigrationPlan:
        return self._localize_scope(service, physical_prediction, scope)

    def _lower_action(self, observation, lower_action_fn: Optional[LowerPolicy] = None) -> int:
        if lower_action_fn is not None:
            return int(lower_action_fn(observation))
        if self.lower_policy is not None:
            return int(self.lower_policy(observation))
        if self.lower_model is None:
            return 0
        with torch.no_grad():
            state = torch.as_tensor(observation.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            candidates = torch.as_tensor(observation.candidate_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.lower_model.act(state, candidates).argmax(dim=-1).item())

    def _failure(
        self,
        service: RunningService,
        node_risk: Dict[int, float],
        link_risk: Dict,
        reason: str,
        plan: Optional[MigrationPlan] = None,
        lower_trace: Optional[List[LowerTransition]] = None,
    ) -> HACExecutionResult:
        risk = self._deployment_risk(service, node_risk, link_risk)
        outcome = ExecutionOutcome(service.deployment.clone(), False, reason, risk, risk, 0, 0, 0.0, 0.0)
        return HACExecutionResult(plan, outcome, len(lower_trace or []), False, False, list(lower_trace or []))

    def execute_now(
        self,
        service: RunningService,
        scope: str,
        physical_prediction: RiskPrediction,
        released_residual_graph: nx.Graph,
        lower_action_fn: Optional[LowerPolicy] = None,
    ) -> HACExecutionResult:
        """Execute using the latest released residual graph; never mutate its owner."""
        expected_node, expected_link = physical_prediction.expected_risk_maps()
        plan = self.execution_plan(service, scope, physical_prediction)
        lower_trace: List[LowerTransition] = []
        try:
            if scope == "link-only":
                outcome = self.executor.execute_with_outcome(
                    TimeVaryingTopology([released_residual_graph]),
                    service,
                    plan,
                    expected_node,
                    expected_link,
                )
                return HACExecutionResult(plan, outcome, 0, True, True, [])
            execution_env = ExecutionEnv()
            observation = execution_env.reset(
                released_residual_graph,
                service,
                plan,
                expected_node,
                physical_prediction.future_node_risk,
            )
            lower_decisions = 0
            while observation is not None:
                action = self._lower_action(observation, lower_action_fn)
                if not 0 <= action < len(observation.candidate_nodes):
                    raise ValueError("Lower policy selected an infeasible candidate index.")
                virtual_node = execution_env.pending_v_nodes[execution_env.cursor]
                physical_node = observation.candidate_nodes[action]
                # Record the sampled action before step() asks for the next VNF.
                # That lookup can fail when no later feasible host remains, but the
                # current policy decision still requires terminal credit.
                lower_trace.append(
                    LowerTransition(
                        observation=observation,
                        action=action,
                        virtual_node=int(virtual_node),
                        physical_node=int(physical_node),
                        done=False,
                    )
                )
                next_observation, done, _ = execution_env.step(action)
                if done:
                    lower_trace[-1] = LowerTransition(
                        observation=observation,
                        action=action,
                        virtual_node=int(virtual_node),
                        physical_node=int(physical_node),
                        done=True,
                    )
                observation = next_observation
                lower_decisions += 1
            if execution_env.graph is None:
                return self._failure(service, expected_node, expected_link, "missing-execution-graph", plan, lower_trace)
            deployment = self.executor.route_links(
                execution_env.graph,
                service,
                execution_env.node_mapping,
                expected_link,
                f"Multi-service HAC reconfiguration ({scope})",
                set(plan.target_v_links),
            )
            outcome = self.executor.assess_deployment(service, deployment, expected_node, expected_link)
            return HACExecutionResult(plan, outcome, lower_decisions, True, True, lower_trace)
        except (RuntimeError, ValueError):
            return self._failure(service, expected_node, expected_link, "lower-or-routing-infeasible", plan, lower_trace)
