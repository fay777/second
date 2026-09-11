"""Evaluate persistence, heuristic, or temporal-only predictors on held-out trajectories."""

import argparse
import csv
import json
from pathlib import Path

import torch

from pe_vnr.prediction_baselines import HeuristicTrendRiskNet, PersistenceRiskNet, TemporalOnlyRiskNet
from pe_vnr.scenarios import ScenarioConfig
from pe_vnr.stgcn_dataset import build_samples, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-graph risk-prediction baselines.")
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", choices=("persistence", "heuristic", "temporal-only"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    reference = torch.load(args.reference_checkpoint, map_location=device, weights_only=True)
    scenario = ScenarioConfig(reference["scenario"], reference["physical_nodes"], reference["virtual_nodes"])
    samples = build_samples(scenario, reference["splits"]["test"], reference["slots"], reference["history_window"], reference["future_horizon"], device)
    if args.baseline == "persistence":
        model = PersistenceRiskNet(reference["future_horizon"]).to(device)
    elif args.baseline == "heuristic":
        model = HeuristicTrendRiskNet(reference["future_horizon"]).to(device)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for temporal-only evaluation.")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = TemporalOnlyRiskNet(reference["node_feature_dim"], reference["link_feature_dim"], checkpoint["hidden_dim"], reference["future_horizon"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate_predictions(model, samples, reference["future_horizon"], reference["risk_threshold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics.keys()); writer.writeheader(); writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
