"""Train the multi-step ST-GCN risk predictor from dynamic topology traces."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from pe_vnr.risk_predictor import STGCNRiskNet
from pe_vnr.scenarios import ScenarioConfig
from pe_vnr.stgcn_dataset import build_samples as build_split_samples, evaluate_predictions, split_trajectory_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-step spatial-temporal topology risk predictor.")
    parser.add_argument("--scenario", choices=("toy", "mobility", "congestion", "outage", "compound", "sagin100"), default="compound")
    parser.add_argument("--physical-nodes", type=int, default=10)
    parser.add_argument("--virtual-nodes", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--risk-threshold", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stgcn"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    scenario_config = ScenarioConfig(args.scenario, args.physical_nodes, args.virtual_nodes)
    splits = split_trajectory_ids(args.seeds, args.train_ratio, args.val_ratio, args.split_seed)
    train_samples = build_split_samples(scenario_config, splits["train"], args.slots, args.window, args.horizon, device)
    val_samples = build_split_samples(scenario_config, splits["val"], args.slots, args.window, args.horizon, device)
    if not train_samples or not val_samples:
        raise RuntimeError("No train/validation samples: increase seeds or slots.")
    first = train_samples[0]
    model = STGCNRiskNet(first.node_input.shape[1], first.link_input.shape[1], args.hidden_dim, args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        model.train()
        for sample in train_samples:
            node_prediction, link_prediction = model(sample.node_input, sample.link_input, sample.adjacency, sample.endpoints)
            loss = functional.mse_loss(node_prediction, sample.node_target) + functional.mse_loss(link_prediction, sample.link_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        train_loss = total_loss / len(train_samples)
        validation = evaluate_predictions(model, val_samples, args.horizon, args.risk_threshold)
        validation_loss = validation["selection_loss"]
        row = {"epoch": epoch + 1, "train_loss": train_loss, **validation}
        history.append(row)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(), "epoch": epoch + 1,
                "best_val_loss": best_validation_loss, "history_window": args.window,
                "future_horizon": args.horizon, "hidden_dim": args.hidden_dim,
                "node_feature_dim": first.node_input.shape[1], "link_feature_dim": first.link_input.shape[1],
                "scenario": args.scenario, "physical_nodes": args.physical_nodes,
                "virtual_nodes": args.virtual_nodes, "slots": args.slots,
                "risk_threshold": args.risk_threshold, "splits": splits,
            }, args.output_dir / "stgcn_best.pt")
        else:
            stale_epochs += 1
        print(f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} val_loss={validation_loss:.6f}")
        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} non-improving epochs.")
            break
    torch.save({"state_dict": model.state_dict(), "history_window": args.window, "future_horizon": args.horizon, "hidden_dim": args.hidden_dim, "node_feature_dim": first.node_input.shape[1], "link_feature_dim": first.link_input.shape[1], "scenario": args.scenario, "physical_nodes": args.physical_nodes, "virtual_nodes": args.virtual_nodes, "slots": args.slots, "risk_threshold": args.risk_threshold, "splits": splits}, args.output_dir / "stgcn_last.pt")
    with (args.output_dir / "validation_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (args.output_dir / "trajectory_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint and validation history to {args.output_dir}")


if __name__ == "__main__":
    main()
