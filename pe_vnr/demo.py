import random

from .config import ExecutionConfig, PlanningConfig, PredictorConfig
from .dynamic_env import DynamicTopologyEnv
from .executor import ElasticReconfigurationExecutor
from .planner import MigrationPlanner
from .risk_predictor import STRiskPredictor
from .scenarios import build_physical_snapshot, build_running_service
from .topology import TimeVaryingTopology


def run_demo() -> None:
    random.seed(7)
    history = [build_physical_snapshot(step) for step in range(4)]
    topology = TimeVaryingTopology(history=history)
    service = build_running_service()

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
        topology_builder=build_physical_snapshot,
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
