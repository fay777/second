"""Train Upper/Lower HAC on the frozen ST-GCN + Top-1 multi-VNR environment."""

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.distributions import Categorical

from pe_vnr.config import ExecutionConfig, PlanningConfig, PredictorConfig
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.hac_worker import MultiServiceHACWorkerAdapter
from pe_vnr.mdp.actor_critic import ExecutionActorCritic, PlanningActorCritic
from pe_vnr.multi_service_env import HACWorkerEvent, MultiServiceConfig, MultiServiceDynamicEnv
from pe_vnr.planner import MigrationPlanner
from pe_vnr.risk_predictor import STRiskPredictor
from pe_vnr.scenarios import DynamicSAGINScenario, ScenarioConfig
from pe_vnr.workload import generate_workload_trace


UPPER_STATE_DIM = 17  # service(5) + risk(4 + horizon=3) + cost/pressure(5)
LOWER_STATE_DIM = 9
LOWER_CANDIDATE_DIM = 7
UPPER_ACTION_DIM = 24  # KEEP/MIGRATE x delay(0..3) x 3 scopes
FUTURE_SLA_HORIZON = 3


def update_policy(model, optimizer, trajectories, gamma: float) -> float:
    """Update from independent trajectories without leaking returns across executions."""
    trajectories = [trajectory for trajectory in trajectories if trajectory]
    if not trajectories:
        return 0.0
    records, return_values = [], []
    for trajectory in trajectories:
        returns, value = [], 0.0
        for record in reversed(trajectory):
            value = float(record["reward"]) + gamma * value
            returns.append(value)
        records.extend(trajectory)
        return_values.extend(reversed(returns))
    returns = torch.tensor(return_values, dtype=torch.float32, device=records[0]["value"].device)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
    values = torch.stack([record["value"] for record in records])
    log_probs = torch.stack([record["log_prob"] for record in records])
    advantage = returns - values.detach()
    loss = -(log_probs * advantage).mean() + 0.5 * torch.nn.functional.mse_loss(values, returns)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def event_reward(event: HACWorkerEvent) -> Tuple[float, Dict[str, float]]:
    """Decompose reward so policy behavior remains auditable in experiments."""
    components = {"risk": 0.0, "cost": 0.0, "disruption": 0.0, "sla": 0.0, "failure": 0.0}
    if event.execution is None:
        return sum(components.values()), components
    outcome = event.execution.outcome
    components["risk"] = 2.0 * float(outcome.risk_reduction)
    components["cost"] = -float(outcome.total_cost)
    if outcome.migrated:
        components["disruption"] = -0.5 if event.decision.scope == "link-only" else -1.0
    else:
        components["failure"] = -1.0
    pre_sla = bool(event.sla_violated)
    post_sla = pre_sla if event.post_sla_violated is None else bool(event.post_sla_violated)
    if pre_sla and not post_sla:
        components["sla"] = 3.0
    elif not pre_sla and post_sla:
        components["sla"] = -3.0
    elif pre_sla:
        components["sla"] = -1.0
    return sum(components.values()), components


def build_environment(seed, steps, checkpoint, device, upper_policy, lower_policy):
    """Create one matched training rollout; ST-GCN and selector remain fixed."""
    scenario = DynamicSAGINScenario(ScenarioConfig("sagin100", 100, 4), seed)
    config = MultiServiceConfig(policy="stgcn_topk_hac")
    executor = ElasticReconfigurationExecutor(ExecutionConfig())
    predictor = STRiskPredictor(PredictorConfig(
        use_learned_model=True,
        checkpoint_path=str(checkpoint),
        require_checkpoint=True,
        device=device,
    ))
    worker = MultiServiceHACWorkerAdapter(
        executor,
        upper_policy=upper_policy,
        lower_policy=lower_policy,
        device=device,
    )
    return MultiServiceDynamicEnv(
        scenario,
        predictor,
        MigrationPlanner(PlanningConfig()),
        executor,
        config,
        workload=generate_workload_trace(scenario, config, 3, steps),
        hac_worker=worker,
    )


def future_keep_penalty(event, service_sla_history):
    """Penalize a KEEP action for observed SLA violations in the ST-GCN horizon."""
    if event.decision.migrate:
        return 0.0
    future = [
        service_sla_history[(time_step, event.service_id)]
        for time_step in range(event.time_step + 1, event.time_step + FUTURE_SLA_HORIZON + 1)
        if (time_step, event.service_id) in service_sla_history
    ]
    return -float(sum(future)) / max(1, len(future))


def assign_event_rewards(events, service_sla_history, upper_records, lower_records):
    """Credit delayed actions by ID and keep each Lower execution independent."""
    upper_events = [event for event in events if event.event_type == "upper_decision"]
    if len(upper_events) != len(upper_records):
        raise RuntimeError("Upper action trace does not match upper-decision events.")
    upper_by_id = {event.origin_decision_id: record for event, record in zip(upper_events, upper_records)}
    lower_queue = deque(lower_records)
    totals = {"risk": 0.0, "cost": 0.0, "disruption": 0.0, "sla": 0.0, "future_sla": 0.0, "failure": 0.0, "total": 0.0}
    lower_trajectories = []
    for event in events:
        reward, components = event_reward(event)
        if event.event_type == "upper_decision" and not event.decision.migrate:
            components["future_sla"] = future_keep_penalty(event, service_sla_history)
            reward += components["future_sla"]
        for name, value in components.items():
            totals[name] += value
        totals["total"] += reward
        upper_record = upper_by_id.get(event.origin_decision_id)
        if upper_record is None:
            raise RuntimeError(f"Missing upper record for HAC decision {event.origin_decision_id}.")
        upper_record["reward"] += reward
        if event.execution is not None:
            execution_records = []
            for _ in event.execution.lower_trace:
                if not lower_queue:
                    raise RuntimeError("Lower action trace does not match HAC execution events.")
                execution_records.append(lower_queue.popleft())
            if execution_records:
                # Only the terminal Lower transition receives the execution reward.
                execution_records[-1]["reward"] += reward
                lower_trajectories.append(execution_records)
    if lower_queue:
        raise RuntimeError("Unassigned lower actions remain after reward assignment.")
    return totals, lower_trajectories


def main():
    parser = argparse.ArgumentParser(description="Train multi-VNR HAC with frozen ST-GCN and Top-1 selector.")
    parser.add_argument("--stgcn-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/multiservice_hac"))
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("episodes and steps must be positive.")
    if not args.stgcn_checkpoint.is_file():
        raise FileNotFoundError(f"Missing ST-GCN checkpoint: {args.stgcn_checkpoint}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    upper = PlanningActorCritic(UPPER_STATE_DIM, UPPER_ACTION_DIM).to(device)
    lower = ExecutionActorCritic(LOWER_STATE_DIM, LOWER_CANDIDATE_DIM).to(device)
    upper_opt = torch.optim.Adam(upper.parameters(), lr=args.lr)
    lower_opt = torch.optim.Adam(lower.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []

    for episode in range(args.episodes):
        upper_records, lower_records = [], []

        def upper_policy(obs):
            state = torch.tensor(obs.state, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(obs.action_mask, dtype=torch.bool, device=device).unsqueeze(0)
            dist = Categorical(logits=upper.act(state).masked_fill(~mask, -1e9))
            action = dist.sample()
            upper_records.append({"log_prob": dist.log_prob(action).squeeze(), "value": upper.value(state).squeeze(), "reward": 0.0})
            return int(action.item())

        def lower_policy(obs):
            state = torch.tensor(obs.state, dtype=torch.float32, device=device).unsqueeze(0)
            candidates = torch.tensor(obs.candidate_features, dtype=torch.float32, device=device).unsqueeze(0)
            dist = Categorical(logits=lower.act(state, candidates))
            action = dist.sample()
            lower_records.append({"log_prob": dist.log_prob(action).squeeze(), "value": lower.value(state).squeeze(), "reward": 0.0})
            return int(action.item())

        seed = args.seed + episode
        env = build_environment(seed, args.steps, args.stgcn_checkpoint, str(device), upper_policy, lower_policy)
        result = env.run(args.steps)
        reward_totals, lower_trajectories = assign_event_rewards(
            result["hac_events"],
            result["service_sla_history"],
            upper_records,
            lower_records,
        )
        upper_loss = update_policy(upper, upper_opt, [upper_records], args.gamma)
        lower_loss = update_policy(lower, lower_opt, lower_trajectories, args.gamma)
        row = {
            "episode": episode,
            "seed": seed,
            "upper_loss": upper_loss,
            "lower_loss": lower_loss,
            "upper_actions": len(upper_records),
            "lower_actions": len(lower_records),
            "lower_executions": len(lower_trajectories),
            "sla_violation_rate": result["sla_violation_rate"],
            "availability": result["availability"],
            "admission_rate": result["admission_rate"],
            "migrations": result["metrics"]["migrations"],
            "total_realized_cost": result["metrics"]["total_realized_cost"],
            "total_disruption": result["metrics"]["total_disruption"],
            **{f"reward_{name}": value for name, value in reward_totals.items()},
        }
        history.append(row)
        print(
            f"episode={episode:03d} upper_loss={upper_loss:.4f} lower_loss={lower_loss:.4f} "
            f"sla={row['sla_violation_rate']:.4f} migrations={row['migrations']} reward={row['reward_total']:.3f}"
        )
    metadata = {
        "stgcn_checkpoint": str(args.stgcn_checkpoint),
        "selector": "TopRiskKSelector",
        "selector_top_k": 1,
        "upper_state_dim": UPPER_STATE_DIM,
        "upper_action_dim": UPPER_ACTION_DIM,
        "lower_state_dim": LOWER_STATE_DIM,
        "lower_candidate_dim": LOWER_CANDIDATE_DIM,
        "reward_components": ["risk", "cost", "disruption", "sla", "future_sla", "failure"],
        "future_sla_horizon": FUTURE_SLA_HORIZON,
        "arguments": {
            **vars(args),
            "stgcn_checkpoint": str(args.stgcn_checkpoint),
            "output": str(args.output),
        },
    }
    torch.save({"state_dict": upper.state_dict(), **metadata}, args.output / "upper.pt")
    torch.save({"state_dict": lower.state_dict(), **metadata}, args.output / "lower.pt")
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
