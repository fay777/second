"""Run a preset or custom composition of predictor, planner, and executor."""

import argparse
import json
from pathlib import Path

from pe_vnr.components import MODE_PRESETS, ComponentConfig, build_environment
from pe_vnr.policy_runner import ModularPolicyRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular proactive VNR experiments.")
    parser.add_argument("--mode", choices=tuple(MODE_PRESETS), default="heuristic-proactive")
    parser.add_argument("--predictor", choices=("heuristic", "stgcn"))
    parser.add_argument("--planner", choices=("heuristic", "ac"))
    parser.add_argument("--execution", choices=("heuristic", "ac"))
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--planning-checkpoint", type=Path)
    parser.add_argument("--execution-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenario", choices=("toy", "mobility", "congestion", "outage", "compound"), default="toy")
    parser.add_argument("--physical-nodes", type=int, default=6)
    parser.add_argument("--virtual-nodes", type=int, default=3)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--future-horizon", type=int, default=3)
    args = parser.parse_args()
    config = ComponentConfig(
        mode=args.mode,
        predictor=args.predictor,
        planner=args.planner,
        execution=args.execution,
        predictor_checkpoint=args.predictor_checkpoint,
        planning_checkpoint=args.planning_checkpoint,
        execution_checkpoint=args.execution_checkpoint,
        seed=args.seed,
        device=args.device,
        scenario=args.scenario,
        physical_nodes=args.physical_nodes,
        virtual_nodes=args.virtual_nodes,
        history_window=args.history_window,
        future_horizon=args.future_horizon,
    )
    components = config.resolved()
    env = build_environment(config)
    if components["planner"] == "heuristic" and components["execution"] == "heuristic":
        result = env.run(args.steps)
    else:
        result = ModularPolicyRunner(
            components["planner"],
            components["execution"],
            config.planning_checkpoint,
            config.execution_checkpoint,
            config.device,
        ).run(env, args.steps)
    print(json.dumps({"mode": args.mode, "components": components, "scenario": args.scenario, "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
