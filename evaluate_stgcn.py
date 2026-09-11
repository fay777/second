"""Evaluate a frozen ST-GCN checkpoint on its trajectory-level test split."""

import argparse
import csv
import json
from pathlib import Path

import torch

from pe_vnr.risk_predictor import STGCNRiskNet
from pe_vnr.scenarios import ScenarioConfig
from pe_vnr.stgcn_dataset import build_samples, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ST-GCN checkpoint on held-out trajectories.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stgcn_test"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    required = {"history_window", "future_horizon", "hidden_dim", "node_feature_dim", "link_feature_dim", "scenario", "physical_nodes", "virtual_nodes", "slots", "risk_threshold", "splits"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint lacks formal-evaluation metadata: {missing}")
    metadata = checkpoint
    scenario = ScenarioConfig(metadata["scenario"], metadata["physical_nodes"], metadata["virtual_nodes"])
    samples = build_samples(
        scenario, metadata["splits"]["test"], metadata["slots"],
        metadata["history_window"], metadata["future_horizon"], device,
    )
    if not samples:
        raise RuntimeError("The checkpoint test split produced no samples.")
    model = STGCNRiskNet(
        metadata["node_feature_dim"], metadata["link_feature_dim"],
        metadata["hidden_dim"], metadata["future_horizon"],
    ).to(device)
    model.load_state_dict(metadata["state_dict"])
    metrics = evaluate_predictions(model, samples, metadata["future_horizon"], metadata["risk_threshold"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
