"""Run a paper-scale shared-resource SAGIN workload without changing HAC."""

import argparse
import json

from pe_vnr.config import ExecutionConfig, PlanningConfig, PredictorConfig
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.multi_service_env import MultiServiceConfig, MultiServiceDynamicEnv
from pe_vnr.planner import MigrationPlanner
from pe_vnr.risk_predictor import STRiskPredictor
from pe_vnr.scenarios import DynamicSAGINScenario, ScenarioConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 100-node multi-VNR SAGIN workload.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arrival-probability", type=float, default=0.45)
    parser.add_argument("--max-active-services", type=int, default=12)
    parser.add_argument("--min-vnfs", type=int, default=2)
    parser.add_argument("--max-vnfs", type=int, default=10)
    args = parser.parse_args()

    scenario = DynamicSAGINScenario(ScenarioConfig("sagin100", 100, 4), args.seed)
    environment = MultiServiceDynamicEnv(
        scenario,
        STRiskPredictor(PredictorConfig(use_learned_model=False, history_window=4, future_horizon=3)),
        MigrationPlanner(PlanningConfig(node_risk_threshold=0.50, link_risk_threshold=0.52, full_migration_threshold=0.72, partial_ratio_threshold=0.34, max_trigger_delay=2)),
        ElasticReconfigurationExecutor(ExecutionConfig()),
        MultiServiceConfig(args.arrival_probability, args.max_active_services, args.min_vnfs, args.max_vnfs),
    )
    result = environment.run(args.steps)
    print(json.dumps({
        "arrivals": result["arrivals"],
        "admissions": result["admissions"],
        "departures": result["departures"],
        "active_services": result["active_services"],
        "metrics": result["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
