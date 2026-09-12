"""Run a paper-scale shared-resource SAGIN workload without changing HAC."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from pe_vnr.config import ExecutionConfig, PlanningConfig, PredictorConfig
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.multi_service_env import MultiServiceConfig, MultiServiceDynamicEnv
from pe_vnr.planner import MigrationPlanner
from pe_vnr.risk_predictor import STRiskPredictor
from pe_vnr.scenarios import DynamicSAGINScenario, ScenarioConfig
from pe_vnr.service_selector import AllRiskySelector, TopRiskKSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 100-node multi-VNR SAGIN workload.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arrival-probability", type=float, default=0.45)
    parser.add_argument("--max-active-services", type=int, default=12)
    parser.add_argument("--min-vnfs", type=int, default=2)
    parser.add_argument("--max-vnfs", type=int, default=10)
    parser.add_argument("--policy", choices=("static", "reactive_full", "proactive_heuristic", "heuristic_all_risky", "stgcn_all_risky", "stgcn_topk_heuristic"), default="proactive_heuristic")
    parser.add_argument("--stgcn-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selector-top-k", type=int, default=1)
    parser.add_argument("--selector-risk-threshold", type=float, default=0.50)
    parser.add_argument("--selector-peak-weight", type=float, default=0.70)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    uses_stgcn = args.policy.startswith("stgcn_")
    if uses_stgcn and not args.stgcn_checkpoint:
        raise ValueError("--stgcn-checkpoint is required for ST-GCN policies.")
    scenario = DynamicSAGINScenario(ScenarioConfig("sagin100", 100, 4), args.seed)
    selector = (
        TopRiskKSelector(args.selector_top_k, args.selector_risk_threshold, args.selector_peak_weight)
        if args.policy == "stgcn_topk_heuristic"
        else AllRiskySelector(args.selector_risk_threshold, args.selector_peak_weight)
    )
    environment = MultiServiceDynamicEnv(
        scenario,
        STRiskPredictor(PredictorConfig(
            use_learned_model=uses_stgcn,
            history_window=4,
            future_horizon=3,
            checkpoint_path=args.stgcn_checkpoint,
            require_checkpoint=uses_stgcn,
            device=args.device,
        )),
        MigrationPlanner(PlanningConfig(node_risk_threshold=0.50, link_risk_threshold=0.52, full_migration_threshold=0.72, partial_ratio_threshold=0.34, max_trigger_delay=2)),
        ElasticReconfigurationExecutor(ExecutionConfig()),
        MultiServiceConfig(args.arrival_probability, args.max_active_services, args.min_vnfs, args.max_vnfs, True, args.policy),
        service_selector=selector,
    )
    result = environment.run(args.steps)
    summary = {
        "policy": args.policy,
        "arrivals": result["arrivals"],
        "admissions": result["admissions"],
        "rejected_admissions": result["rejected_admissions"],
        "departures": result["departures"],
        "active_services": result["active_services"],
        "sla_violation_rate": result["sla_violation_rate"],
        "availability": result["availability"],
        "avg_end_to_end_delay": result["avg_end_to_end_delay"],
        "selector_candidate_count": result["selector_candidate_count"],
        "selector_selected_count": result["selector_selected_count"],
        "selector_pending_count": result["selector_pending_count"],
        "avg_active_pending_plans": result["avg_active_pending_plans"],
        "max_active_pending_plans": result["max_active_pending_plans"],
        "metrics": result["metrics"],
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for filename, records in {
            "step_metrics.csv": result["steps"],
            "selection_metrics.csv": result["selection_records"],
        }.items():
            if not records:
                continue
            with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=asdict(records[0]).keys())
                writer.writeheader()
                writer.writerows(asdict(record) for record in records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
