"""Train a temporal-only predictor on the exact ST-GCN trajectory split."""

import argparse
import torch
import torch.nn.functional as functional
from pathlib import Path

from pe_vnr.prediction_baselines import TemporalOnlyRiskNet
from pe_vnr.scenarios import ScenarioConfig
from pe_vnr.stgcn_dataset import build_samples, evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the non-graph temporal prediction baseline.")
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/temporal_only_best.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.reference_checkpoint, map_location=device, weights_only=True)
    metadata = checkpoint
    scenario = ScenarioConfig(metadata["scenario"], metadata["physical_nodes"], metadata["virtual_nodes"])
    train_samples = build_samples(scenario, metadata["splits"]["train"], metadata["slots"], metadata["history_window"], metadata["future_horizon"], device)
    val_samples = build_samples(scenario, metadata["splits"]["val"], metadata["slots"], metadata["history_window"], metadata["future_horizon"], device)
    model = TemporalOnlyRiskNet(metadata["node_feature_dim"], metadata["link_feature_dim"], args.hidden_dim, metadata["future_horizon"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, stale = float("inf"), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for sample in train_samples:
            node_prediction, link_prediction = model(sample.node_input, sample.link_input)
            loss = functional.mse_loss(node_prediction, sample.node_target) + functional.mse_loss(link_prediction, sample.link_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        validation = evaluate_predictions(model, val_samples, metadata["future_horizon"], metadata["risk_threshold"])
        if validation["selection_loss"] < best:
            best, stale = validation["selection_loss"], 0
            torch.save({"state_dict": model.state_dict(), "model_type": "temporal-only", "hidden_dim": args.hidden_dim, **{key: metadata[key] for key in ("history_window", "future_horizon", "node_feature_dim", "link_feature_dim", "scenario", "physical_nodes", "virtual_nodes", "slots", "risk_threshold", "splits")}}, args.output)
        else:
            stale += 1
        print(f"epoch={epoch + 1:03d} val_loss={validation['selection_loss']:.6f}")
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
