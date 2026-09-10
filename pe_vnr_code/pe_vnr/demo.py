import random

import networkx as nx

from .config import ExecutionConfig, PlanningConfig, PredictorConfig
from .dynamic_env import DynamicTopologyEnv
from .executor import ElasticReconfigurationExecutor
from .planner import MigrationPlanner
from .risk_predictor import STRiskPredictor
from .topology import Deployment, RunningService, TimeVaryingTopology, edge_key


def _build_physical_snapshot(step: int) -> nx.Graph:
    graph = nx.Graph()
    for node_id in range(6):
        graph.add_node(
            node_id,
            cpu=max(8.0, 88.0 - 8.0 * node_id - 5.0 * step + random.uniform(-4.0, 4.0)),
            max_cpu=100.0,
            storage=50.0,
            energy=max(0.05, 1.0 - 0.08 * step - 0.03 * node_id),
            queue=min(1.0, 0.18 * step + 0.06 * node_id),
            domain=0 if node_id < 2 else (1 if node_id < 4 else 2),
        )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
    ]
    for u, v in edges:
        graph.add_edge(
            u,
            v,
            bandwidth=max(8.0, 92.0 - 7.0 * step - 2.5 * (u + v) + random.uniform(-6.0, 6.0)),
            max_bandwidth=100.0,
            delay=2.0 + abs(u - v) + 0.35 * step,
            loss=min(1.0, 0.03 * step + 0.015 * abs(u - v)),
            visible_time=max(0.2, 1.5 - 0.14 * step - 0.08 * abs(u - v)),
        )
    return graph


def _build_running_service() -> RunningService:
    virtual_graph = nx.Graph()
    virtual_graph.add_node(0, cpu=18.0)
    virtual_graph.add_node(1, cpu=22.0)
    virtual_graph.add_node(2, cpu=16.0)
    virtual_graph.add_edge(0, 1, bandwidth=18.0)
    virtual_graph.add_edge(1, 2, bandwidth=15.0)

    deployment = Deployment(
        node_mapping={0: 1, 1: 2, 2: 3},
        link_mapping={
            (0, 1): [edge_key(1, 2)],
            (1, 2): [edge_key(2, 3)],
        },
        description="Current deployment",
    )
    return RunningService(
        service_id="svc-001",
        virtual_graph=virtual_graph,
        deployment=deployment,
        remaining_lifetime=8,
        max_delay=12.0,
        priority=1.0,
    )


def run_demo() -> None:
    random.seed(7)
    history = [_build_physical_snapshot(step) for step in range(4)]
    topology = TimeVaryingTopology(history=history)
    service = _build_running_service()

    predictor = STRiskPredictor(PredictorConfig(use_learned_model=False, history_window=4, future_horizon=3))
    planner = MigrationPlanner(
        PlanningConfig(
            node_risk_threshold=0.50,
            link_risk_threshold=0.52,
            full_migration_threshold=0.72,
            partial_ratio_threshold=0.34,
            max_trigger_delay=2,
        )
    )
    executor = ElasticReconfigurationExecutor(ExecutionConfig())
    env = DynamicTopologyEnv(
        topology=topology,
        service=service,
        topology_builder=_build_physical_snapshot,
        predictor=predictor,
        planner=planner,
        executor=executor,
    )

    results = env.run(num_steps=4)

    print("=== Step-by-step Reconfiguration ===")
    start_time = env.time_step - len(results["steps"]) + 1
    for offset, step_result in enumerate(results["steps"]):
        print(f"[t={start_time + offset}]")
        print(f"Plan migrate  : {step_result.plan.migrate}")
        print(f"Trigger delay : {step_result.plan.trigger_delay}")
        print(f"Scope         : {step_result.plan.scope}")
        print(f"Risk score    : {step_result.plan.risk_score:.4f}")
        print(f"Estimated cost: {step_result.plan.estimated_cost:.4f}")
        print(f"Realized cost : {step_result.realized_cost:.4f}")
        print(f"Post risk     : {step_result.post_risk:.4f}")
        print(f"SLA violated  : {step_result.sla_violated}")
        print(f"Node mapping  : {step_result.service.deployment.node_mapping}")
        print()

    final_service = results["service"]
    print("=== Final Service State ===")
    print(f"Status          : {final_service.status}")
    print(f"Remaining life  : {final_service.remaining_lifetime}")
    print(f"Migration count : {final_service.migration_count}")
    print(f"Disruption time : {final_service.disruption_time:.2f}")
    print(f"SLA violations  : {final_service.sla_violations}")
    print(f"Deployment      : {final_service.deployment.node_mapping}")
    print()

    print("=== Metrics Summary ===")
    for metric_name, metric_value in results["metrics"].items():
        print(f"{metric_name:18}: {metric_value:.4f}")
