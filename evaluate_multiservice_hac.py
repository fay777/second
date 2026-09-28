"""Evaluate a saved multi-VNR HAC checkpoint on held-out matched seeds."""

import argparse
import json
from pathlib import Path

import torch

from pe_vnr.mdp.actor_critic import ExecutionActorCritic, PlanningActorCritic
from train_multiservice_hac import (
    LOWER_CANDIDATE_DIM,
    LOWER_STATE_DIM,
    UPPER_ACTION_DIM,
    UPPER_STATE_DIM,
    evaluate_argmax_policy,
    evaluate_static_policy,
    summarize_validation,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved multi-VNR HAC checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stgcn-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-prefix", choices=("best", "final"), default="best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.stgcn_checkpoint.is_file():
        raise FileNotFoundError(f"Missing ST-GCN checkpoint: {args.stgcn_checkpoint}")
    prefix = "best" if args.checkpoint_prefix == "best" else ""
    upper_path = args.checkpoint_dir / f"{prefix + '_' if prefix else ''}upper.pt"
    lower_path = args.checkpoint_dir / f"{prefix + '_' if prefix else ''}lower.pt"
    if not upper_path.is_file() or not lower_path.is_file():
        raise FileNotFoundError(f"Missing HAC checkpoint pair: {upper_path}, {lower_path}")

    device = torch.device(args.device)
    upper_payload = torch.load(upper_path, map_location=device)
    lower_payload = torch.load(lower_path, map_location=device)
    upper = PlanningActorCritic(UPPER_STATE_DIM, UPPER_ACTION_DIM).to(device)
    lower = ExecutionActorCritic(LOWER_STATE_DIM, LOWER_CANDIDATE_DIM).to(device)
    upper.load_state_dict(upper_payload["state_dict"])
    lower.load_state_dict(lower_payload["state_dict"])

    hac_rows = [
        evaluate_argmax_policy(upper, lower, seed, args.steps, args.stgcn_checkpoint, device)
        for seed in args.seeds
    ]
    static_rows = [evaluate_static_policy(seed, args.steps) for seed in args.seeds]
    summary = summarize_validation(hac_rows, episode=0)
    static_sla = sum(row["sla_violation_rate"] for row in static_rows) / len(static_rows)
    summary["static_sla_violation_rate"] = static_sla
    summary["sla_improvement_over_static"] = static_sla - summary["val_sla_violation_rate"]

    output = args.output or args.checkpoint_dir / f"{args.checkpoint_prefix}_development_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint_prefix": args.checkpoint_prefix,
                "upper_checkpoint": str(upper_path),
                "lower_checkpoint": str(lower_path),
                "stgcn_checkpoint": str(args.stgcn_checkpoint),
                "seeds": args.seeds,
                "steps": args.steps,
                "summary": summary,
                "hac": hac_rows,
                "static": static_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"saved evaluation: {output}")


if __name__ == "__main__":
    main()
