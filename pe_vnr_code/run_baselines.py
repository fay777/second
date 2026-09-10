"""Run reproducible baselines for proactive elastic VNR experiments.

The script compares policies on identical time-varying SAGIN trajectories.  It
is deliberately dependency-light so that CSV files can be used directly by
Excel, Origin, or a later plotting notebook.
"""

import argparse
import csv
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

from pe_vnr.config import ExecutionConfig, PlanningConfig, PredictorConfig
from pe_vnr.demo import _build_physical_snapshot, _build_running_service
from pe_vnr.dynamic_env import DynamicTopologyEnv
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.planner import MigrationPlan, MigrationPlanner
from pe_vnr.risk_predictor import STRiskPredictor
from pe_vnr.topology import RiskPrediction, RunningService, TimeVaryingTopology, edge_key


class NoReconfigurationPlanner:
    """Static deployment baseline: it observes risks but never intervenes."""

    def __init__(self, delegate: MigrationPlanner):
        self.delegate = delegate
        self.config = delegate.config

    def plan(self, service: RunningService, prediction: RiskPrediction) -> MigrationPlan:
        observed = self.delegate.plan(service, prediction)
        return MigrationPlan(False, 0, "none", [], [], 0.0, observed.risk_score)


class ReactiveMigrationPlanner:
    """Migrate only after the current deployment has violated its delay SLA."""

    def __init__(self, delegate: MigrationPlanner, topology: TimeVaryingTopology):
        self.delegate = delegate
        self.topology = topology
        self.config = delegate.config

    def _delay(self, service: RunningService) -> float:
        delay = 0.0
        for path in service.deployment.link_mapping.values():
            for link_id in path:
                u, v = edge_key(*link_id)
                if self.topology.current.has_edge(u, v):
                    delay += float(self.topology.current.edges[(u, v)].get("delay", 0.0))
        return delay

    def plan(self, service: RunningService, prediction: RiskPrediction) -> MigrationPlan:
        observed = self.delegate.plan(service, prediction)
        if self._delay(service) <= service.max_delay:
            return MigrationPlan(False, 0, "none", [], [], 0.0, observed.risk_score)
        return MigrationPlan(
            True,
            0,
            "full",
            list(service.virtual_graph.nodes),
            list(service.virtual_graph.edges),
            1.0,
            observed.risk_score,
        )


def build_environment(seed: int, baseline: str) -> DynamicTopologyEnv:
    random.seed(seed)
    history = [_build_physical_snapshot(step) for step in range(4)]
    topology = TimeVaryingTopology(history=history)
    config = PlanningConfig(
        node_risk_threshold=0.50,
        link_risk_threshold=0.52,
        full_migration_threshold=0.72,
        partial_ratio_threshold=0.34,
        max_trigger_delay=2,
    )
    proactive = MigrationPlanner(config)
    if baseline == "no_reconfiguration":
        planner = NoReconfigurationPlanner(proactive)
    elif baseline == "reactive":
        planner = ReactiveMigrationPlanner(proactive, topology)
    elif baseline == "proactive_heuristic":
        planner = proactive
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return DynamicTopologyEnv(
        topology=topology,
        service=_build_running_service(),
        topology_builder=_build_physical_snapshot,
        # Do not compare policies using a randomly initialized neural predictor.
        predictor=STRiskPredictor(PredictorConfig(use_learned_model=False, history_window=4, future_horizon=3)),
        planner=planner,
        executor=ElasticReconfigurationExecutor(ExecutionConfig()),
    )


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VNR reconfiguration baselines on matched topology trajectories.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of independent random seeds.")
    parser.add_argument("--steps", type=int, default=6, help="Maximum time slots per episode.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for CSV artifacts.")
    args = parser.parse_args()
    if args.seeds < 2:
        raise ValueError("Use at least two seeds to report a meaningful mean and standard deviation.")

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_names = ("no_reconfiguration", "reactive", "proactive_heuristic")
    rows: List[Dict[str, object]] = []
    for seed in range(args.seeds):
        for baseline in baseline_names:
            environment = build_environment(seed, baseline)
            result = environment.run(args.steps)
            rows.append({"seed": seed, "baseline": baseline, **result["metrics"]})

    metric_names = [name for name in rows[0] if name not in {"seed", "baseline"}]
    summary_rows: List[Dict[str, object]] = []
    for baseline in baseline_names:
        selected = [row for row in rows if row["baseline"] == baseline]
        summary: Dict[str, object] = {"baseline": baseline, "runs": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values)
        summary_rows.append(summary)

    write_csv(args.output / "baseline_runs.csv", rows, ["seed", "baseline", *metric_names])
    write_csv(args.output / "baseline_summary.csv", summary_rows, list(summary_rows[0]))
    print(f"Wrote {args.output / 'baseline_runs.csv'}")
    print(f"Wrote {args.output / 'baseline_summary.csv'}")
    for row in summary_rows:
        print(
            f"{row['baseline']:22} "
            f"post-risk={row['avg_post_risk_mean']:.4f} "
            f"SLA={row['sla_violations_mean']:.2f} "
            f"migrations={row['migrations_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
