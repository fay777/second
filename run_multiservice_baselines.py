"""Run Static, Reactive-Full, and Proactive-Heuristic on matched SAGIN workloads."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from pe_vnr.config import ExecutionConfig, PlanningConfig, PredictorConfig
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.multi_service_env import MultiServiceConfig, MultiServiceDynamicEnv
from pe_vnr.planner import MigrationPlanner
from pe_vnr.risk_predictor import STRiskPredictor
from pe_vnr.scenarios import DynamicSAGINScenario, ScenarioConfig
from pe_vnr.workload import generate_workload_trace


def run_seed(seed: int, policy: str, args) -> dict:
    scenario = DynamicSAGINScenario(ScenarioConfig("sagin100", 100, 4), seed)
    workload_config = MultiServiceConfig(args.arrival_probability, args.max_active_services, args.min_vnfs, args.max_vnfs)
    workload = generate_workload_trace(scenario, workload_config, start_time=3, steps=args.steps)
    environment = MultiServiceDynamicEnv(
        scenario,
        STRiskPredictor(PredictorConfig(use_learned_model=False, history_window=4, future_horizon=3)),
        MigrationPlanner(PlanningConfig(node_risk_threshold=0.50, link_risk_threshold=0.52, full_migration_threshold=0.72, partial_ratio_threshold=0.34, max_trigger_delay=2)),
        ElasticReconfigurationExecutor(ExecutionConfig()),
        MultiServiceConfig(args.arrival_probability, args.max_active_services, args.min_vnfs, args.max_vnfs, True, policy),
        workload=workload,
    )
    result = environment.run(args.steps)
    environment_metrics = {key: value for key, value in result.items() if key not in {"steps", "services", "metrics"}}
    reconfiguration_metrics = {
        key if key not in environment_metrics else f"reconfig_{key}": value
        for key, value in result["metrics"].items()
    }
    return {"seed": seed, "baseline": policy, **environment_metrics, **reconfiguration_metrics}


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched multi-VNR SAGIN baselines.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--arrival-probability", type=float, default=0.45)
    parser.add_argument("--max-active-services", type=int, default=20)
    parser.add_argument("--min-vnfs", type=int, default=2)
    parser.add_argument("--max-vnfs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multiservice_baselines"))
    args = parser.parse_args()
    if args.seeds < 2:
        raise ValueError("Use at least two seeds to report mean and standard deviation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = ("static", "reactive_full", "proactive_heuristic")
    rows = [run_seed(seed, policy, args) for seed in range(args.seeds) for policy in policies]
    metric_names = [name for name in rows[0] if name not in {"seed", "baseline"}]
    summaries = []
    for policy in policies:
        selected = [row for row in rows if row["baseline"] == policy]
        summary = {"baseline": policy, "runs": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values)
        summaries.append(summary)
    write_csv(args.output_dir / "baseline_runs.csv", rows)
    write_csv(args.output_dir / "baseline_summary.csv", summaries)
    for summary in summaries:
        print(f"{summary['baseline']:22} SLA={summary['sla_violation_rate_mean']:.4f} availability={summary['availability_mean']:.4f} admission={summary['admission_rate_mean']:.4f}")


if __name__ == "__main__":
    main()
