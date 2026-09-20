"""Train Upper/Lower HAC on the frozen ST-GCN + Top-1 multi-VNR environment."""

import argparse
import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
UPPER_PLANNING_ACTION_DIM = 24  # Existing PlanningEnv encoding.
UPPER_MIGRATION_ACTION_DIM = 12  # 4 delays x 3 scopes.
UPPER_ACTION_DIM = 2 + UPPER_MIGRATION_ACTION_DIM  # Binary decision + conditional migration action.
UPPER_MIGRATION_OFFSET = UPPER_PLANNING_ACTION_DIM - UPPER_MIGRATION_ACTION_DIM
FUTURE_SLA_HORIZON = 3
REWARD_COMPONENTS = ("risk", "cost", "disruption", "sla", "future_sla", "failure")
CHECKPOINT_EPSILON = 1e-6


@dataclass(frozen=True)
class RewardWeights:
    """Explicit reward weights; defaults reproduce the prior reward definition."""

    risk: float = 2.0
    cost: float = 1.0
    disruption: float = 1.0
    sla: float = 1.0
    future_sla: float = 1.0
    failure: float = 1.0


def migration_action_mask(planning_mask, device) -> torch.Tensor:
    """Extract feasibility mask for the conditional delay/scope migration policy."""
    return torch.tensor(
        [bool(value) for value in planning_mask[UPPER_MIGRATION_OFFSET:]],
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)


def migration_action_to_planning_action(action: int) -> int:
    """Map conditional migration action 0..11 to PlanningEnv action 12..23."""
    if not 0 <= action < UPPER_MIGRATION_ACTION_DIM:
        raise ValueError(f"Conditional migration action is outside action space: {action}")
    return action + UPPER_MIGRATION_OFFSET


def upper_action_distributions(model, state, planning_mask, device):
    """Build a binary KEEP/MIGRATE policy and its conditional migration policy."""
    logits = model.act(state)
    migrate_mask = migration_action_mask(planning_mask, device)
    migrate_available = bool(migrate_mask.any().item())
    decision_mask = torch.tensor(
        [[True, migrate_available]], dtype=torch.bool, device=device
    )
    decision_logits = logits[:, :2].masked_fill(~decision_mask, -1e9)
    decision_dist = Categorical(logits=decision_logits)
    if not migrate_available:
        return decision_dist, None, decision_logits
    migration_logits = logits[:, 2:].masked_fill(~migrate_mask, -1e9)
    return decision_dist, Categorical(logits=migration_logits), decision_logits


def update_policy(model, optimizer, trajectories, gamma: float) -> float:
    """Update from independent trajectories using raw critic targets."""
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
    values = torch.stack([record["value"] for record in records])
    log_probs = torch.stack([record["log_prob"] for record in records])
    advantages = returns - values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = torch.nn.functional.mse_loss(values, returns)
    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def weighted_reward(raw_components: Dict[str, float], weights: RewardWeights) -> Tuple[float, Dict[str, float]]:
    """Apply calibration weights without obscuring the underlying environment signal."""
    weighted = {name: float(raw_components.get(name, 0.0)) * getattr(weights, name) for name in REWARD_COMPONENTS}
    return sum(weighted.values()), weighted


def event_reward(event: HACWorkerEvent, weights: RewardWeights) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Return raw and calibrated reward components for reproducible experiments."""
    components = {name: 0.0 for name in REWARD_COMPONENTS}
    if event.execution is None:
        reward, weighted = weighted_reward(components, weights)
        return reward, components, weighted
    outcome = event.execution.outcome
    # Global max-risk can mask a beneficial link-only reroute when node risk dominates.
    # HAC therefore optimizes the additive node/link exposure reduction, while the
    # legacy global max-risk remains the shared reporting metric for all methods.
    components["risk"] = float(outcome.node_risk_reduction + outcome.link_risk_reduction)
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
    reward, weighted = weighted_reward(components, weights)
    return reward, components, weighted


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


def evaluate_static_policy(seed, steps):
    """Compute the no-reconfiguration reference on a held-out matched workload."""
    scenario = DynamicSAGINScenario(ScenarioConfig("sagin100", 100, 4), seed)
    config = MultiServiceConfig(policy="static")
    executor = ElasticReconfigurationExecutor(ExecutionConfig())
    env = MultiServiceDynamicEnv(
        scenario,
        None,
        MigrationPlanner(PlanningConfig()),
        executor,
        config,
        workload=generate_workload_trace(scenario, config, 3, steps),
    )
    result = env.run(steps)
    return {
        "seed": seed,
        "sla_violation_rate": result["sla_violation_rate"],
        "availability": result["availability"],
        "admission_rate": result["admission_rate"],
    }


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


def assign_event_rewards(events, service_sla_history, upper_records, lower_records, weights: RewardWeights):
    """Credit delayed actions by ID and keep each Lower execution independent."""
    upper_events = [event for event in events if event.event_type == "upper_decision"]
    if len(upper_events) != len(upper_records):
        raise RuntimeError("Upper action trace does not match upper-decision events.")
    upper_by_id = {event.origin_decision_id: record for event, record in zip(upper_events, upper_records)}
    lower_queue = deque(lower_records)
    raw_totals = {name: 0.0 for name in REWARD_COMPONENTS}
    weighted_totals = {name: 0.0 for name in REWARD_COMPONENTS}
    weighted_totals["total"] = 0.0
    lower_trajectories = []
    for event in events:
        reward, components, weighted_components = event_reward(event, weights)
        if event.event_type == "upper_decision" and not event.decision.migrate:
            components["future_sla"] = future_keep_penalty(event, service_sla_history)
            reward, weighted_components = weighted_reward(components, weights)
        for name, value in components.items():
            raw_totals[name] += value
        for name, value in weighted_components.items():
            weighted_totals[name] += value
        weighted_totals["total"] += reward
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
    raw_totals["total"] = sum(raw_totals.values())
    return raw_totals, weighted_totals, lower_trajectories


def execution_diagnostics(execution_events):
    """Summarize execution-only quality without mixing in Upper KEEP credits."""
    scopes = ("link-only", "partial", "full")
    accepted_events = [event for event in execution_events if event.execution.outcome.migrated]
    diagnostics = {
        "execution_events": len(execution_events),
        "accepted_migrations": len(accepted_events),
    }
    for scope in scopes:
        attempts = [event for event in execution_events if event.decision.scope == scope]
        accepted = [event for event in accepted_events if event.decision.scope == scope]
        deltas = [float(event.execution.outcome.risk_reduction) for event in accepted]
        node_deltas = [float(event.execution.outcome.node_risk_reduction) for event in accepted]
        link_deltas = [float(event.execution.outcome.link_risk_reduction) for event in accepted]
        diagnostics[f"execution_scope_{scope}_attempts"] = len(attempts)
        diagnostics[f"accepted_scope_{scope}"] = len(accepted)
        diagnostics[f"accepted_rate_scope_{scope}"] = len(accepted) / max(1, len(attempts))
        diagnostics[f"accepted_avg_risk_reduction_scope_{scope}"] = sum(deltas) / max(1, len(deltas))
        diagnostics[f"accepted_avg_node_risk_reduction_scope_{scope}"] = sum(node_deltas) / max(1, len(node_deltas))
        diagnostics[f"accepted_avg_link_risk_reduction_scope_{scope}"] = sum(link_deltas) / max(1, len(link_deltas))
        diagnostics[f"accepted_risk_improved_scope_{scope}"] = sum(delta > 1e-9 for delta in deltas)
        diagnostics[f"accepted_risk_worsened_scope_{scope}"] = sum(delta < -1e-9 for delta in deltas)
        diagnostics[f"accepted_risk_unchanged_scope_{scope}"] = sum(abs(delta) <= 1e-9 for delta in deltas)
    deltas = [float(event.execution.outcome.risk_reduction) for event in accepted_events]
    node_deltas = [float(event.execution.outcome.node_risk_reduction) for event in accepted_events]
    link_deltas = [float(event.execution.outcome.link_risk_reduction) for event in accepted_events]
    positive = [delta for delta in deltas if delta > 1e-9]
    negative = [delta for delta in deltas if delta < -1e-9]
    diagnostics["accepted_avg_risk_reduction"] = sum(deltas) / max(1, len(deltas))
    diagnostics["accepted_avg_node_risk_reduction"] = sum(node_deltas) / max(1, len(node_deltas))
    diagnostics["accepted_avg_link_risk_reduction"] = sum(link_deltas) / max(1, len(link_deltas))
    diagnostics["accepted_risk_improved_count"] = len(positive)
    diagnostics["accepted_risk_worsened_count"] = len(negative)
    diagnostics["accepted_risk_unchanged_count"] = sum(abs(delta) <= 1e-9 for delta in deltas)
    diagnostics["accepted_risk_improved_fraction"] = len(positive) / max(1, len(deltas))
    diagnostics["accepted_risk_worsened_fraction"] = len(negative) / max(1, len(deltas))
    diagnostics["accepted_risk_unchanged_fraction"] = diagnostics["accepted_risk_unchanged_count"] / max(1, len(deltas))
    diagnostics["accepted_avg_positive_risk_reduction"] = sum(positive) / max(1, len(positive))
    diagnostics["accepted_avg_negative_risk_reduction"] = sum(negative) / max(1, len(negative))
    diagnostics["accepted_avg_realized_cost"] = sum(event.execution.outcome.total_cost for event in accepted_events) / max(1, len(accepted_events))
    diagnostics["accepted_avg_disruption"] = sum(
        0.5 if event.decision.scope == "link-only" else 1.0 for event in accepted_events
    ) / max(1, len(accepted_events))
    return diagnostics


def evaluate_argmax_policy(upper, lower, seed, steps, checkpoint, device):
    """Run a fixed-seed deterministic rollout solely for reward-calibration comparison."""
    upper.eval()
    lower.eval()
    keep_probabilities = []
    migration_margins = []

    def upper_policy(obs):
        state = torch.tensor(obs.state, dtype=torch.float32, device=device).unsqueeze(0)
        decision_dist, migration_dist, decision_logits = upper_action_distributions(
            upper, state, obs.action_mask, device
        )
        probabilities = torch.softmax(decision_logits, dim=-1)
        keep_probabilities.append(float(probabilities[0, 0].item()))
        if migration_dist is not None:
            migration_margins.append(float((decision_logits[0, 1] - decision_logits[0, 0]).item()))
        decision = int(decision_logits.argmax(dim=-1).item())
        if decision == 0:
            return 0
        return migration_action_to_planning_action(int(migration_dist.logits.argmax(dim=-1).item()))

    def lower_policy(obs):
        state = torch.tensor(obs.state, dtype=torch.float32, device=device).unsqueeze(0)
        candidates = torch.tensor(obs.candidate_features, dtype=torch.float32, device=device).unsqueeze(0)
        return int(lower.act(state, candidates).argmax(dim=-1).item())

    with torch.inference_mode():
        env = build_environment(seed, steps, checkpoint, str(device), upper_policy, lower_policy)
        result = env.run(steps)
    execution_events = [event for event in result["hac_events"] if event.execution is not None]
    diagnostics = execution_diagnostics(execution_events)
    stage_counts = Counter(event.execution.failure_stage for event in execution_events)
    return {
        "seed": seed,
        "policy": "argmax",
        "sla_violation_rate": result["sla_violation_rate"],
        "availability": result["availability"],
        "admission_rate": result["admission_rate"],
        "migrations": result["metrics"]["migrations"],
        "total_realized_cost": result["metrics"]["total_realized_cost"],
        "total_disruption": result["metrics"]["total_disruption"],
        "rollback_count": result["rollback_count"],
        "upper_keep_count": result["upper_keep_count"],
        "upper_actions": sum(event.event_type == "upper_decision" for event in result["hac_events"]),
        "keep_rate": result["upper_keep_count"] / max(1, sum(event.event_type == "upper_decision" for event in result["hac_events"])),
        "mean_keep_probability": sum(keep_probabilities) / max(1, len(keep_probabilities)),
        "mean_migration_vs_keep_logit_margin": sum(migration_margins) / max(1, len(migration_margins)),
        "no_candidate_host_rate": stage_counts["no_candidate_host"] / max(1, len(execution_events)),
        "execution_stages": dict(sorted(stage_counts.items())),
        **diagnostics,
    }


def summarize_validation(evaluations, episode: int) -> Dict[str, float]:
    """Aggregate fixed-seed argmax validation without mixing seed-level ratios."""
    accepted = sum(row["accepted_migrations"] for row in evaluations)
    execution_events = sum(row["execution_events"] for row in evaluations)
    upper_actions = sum(row["upper_actions"] for row in evaluations)
    keep_count = sum(row["upper_keep_count"] for row in evaluations)
    weighted = lambda key: sum(row["accepted_migrations"] * row[key] for row in evaluations) / max(1, accepted)
    return {
        "episode": episode,
        "validation_seeds": [row["seed"] for row in evaluations],
        "val_sla_violation_rate": sum(row["sla_violation_rate"] for row in evaluations) / len(evaluations),
        "val_availability": sum(row["availability"] for row in evaluations) / len(evaluations),
        "val_admission_rate": sum(row["admission_rate"] for row in evaluations) / len(evaluations),
        "val_migrations": sum(row["migrations"] for row in evaluations) / len(evaluations),
        "val_total_realized_cost": sum(row["total_realized_cost"] for row in evaluations) / len(evaluations),
        "val_total_disruption": sum(row["total_disruption"] for row in evaluations) / len(evaluations),
        "val_execution_events": execution_events,
        "val_accepted_migrations": accepted,
        "val_execution_success_rate": accepted / max(1, execution_events),
        "val_keep_rate": keep_count / max(1, upper_actions),
        "val_accepted_avg_risk_reduction": weighted("accepted_avg_risk_reduction"),
        "val_accepted_avg_node_risk_reduction": weighted("accepted_avg_node_risk_reduction"),
        "val_accepted_avg_link_risk_reduction": weighted("accepted_avg_link_risk_reduction"),
        "val_risk_improved_fraction": sum(
            row["accepted_risk_improved_count"] for row in evaluations
        ) / max(1, accepted),
        "val_risk_worsened_fraction": sum(
            row["accepted_risk_worsened_count"] for row in evaluations
        ) / max(1, accepted),
        "val_accepted_avg_realized_cost": weighted("accepted_avg_realized_cost"),
        "val_accepted_avg_disruption": weighted("accepted_avg_disruption"),
        "val_rollback_rate": sum(row["rollback_count"] for row in evaluations) / max(1, execution_events),
    }


def is_better_checkpoint(
    candidate: Dict[str, float],
    best: Optional[Dict[str, float]],
    static_sla_violation_rate: float,
) -> bool:
    """Select only proactive improvements over Static, then rank by SLA and cost."""
    if (
        candidate["val_accepted_migrations"] == 0
        or candidate["val_accepted_avg_risk_reduction"] <= CHECKPOINT_EPSILON
        or candidate["val_sla_violation_rate"] >= static_sla_violation_rate - CHECKPOINT_EPSILON
    ):
        return False
    if best is None:
        return True
    for metric in (
        "val_sla_violation_rate",
        "val_total_realized_cost",
        "val_total_disruption",
        "val_migrations",
    ):
        if candidate[metric] < best[metric] - 1e-9:
            return True
        if candidate[metric] > best[metric] + 1e-9:
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Train multi-VNR HAC with frozen ST-GCN and Top-1 selector.")
    parser.add_argument("--stgcn-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-risk-weight", type=float, default=2.0)
    parser.add_argument("--reward-cost-weight", type=float, default=1.0)
    parser.add_argument("--reward-disruption-weight", type=float, default=1.0)
    parser.add_argument("--reward-sla-weight", type=float, default=1.0)
    parser.add_argument("--reward-future-sla-weight", type=float, default=1.0)
    parser.add_argument("--reward-failure-weight", type=float, default=1.0)
    parser.add_argument(
        "--calibration-eval-seeds",
        type=int,
        nargs="*",
        default=[900, 901, 902],
        help="Fixed seeds for post-training argmax calibration rollout; pass with no values to disable.",
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Held-out seeds for periodic argmax validation; pass with no values to disable.",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10,
        help="Run validation every N completed training episodes.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostic", action="store_true", help="Print HAC action and reward-component diagnostics per episode.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/multiservice_hac"))
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("episodes and steps must be positive.")
    if args.validation_interval < 1:
        raise ValueError("validation_interval must be positive.")
    if not args.stgcn_checkpoint.is_file():
        raise FileNotFoundError(f"Missing ST-GCN checkpoint: {args.stgcn_checkpoint}")
    training_seeds = set(range(args.seed, args.seed + args.episodes))
    validation_seeds = set(args.validation_seeds)
    calibration_seeds = set(args.calibration_eval_seeds)
    if training_seeds & validation_seeds:
        raise ValueError("validation_seeds must not overlap with per-episode training seeds.")
    if validation_seeds & calibration_seeds:
        raise ValueError("validation_seeds must not overlap with calibration_eval_seeds.")
    reward_weights = RewardWeights(
        risk=args.reward_risk_weight,
        cost=args.reward_cost_weight,
        disruption=args.reward_disruption_weight,
        sla=args.reward_sla_weight,
        future_sla=args.reward_future_sla_weight,
        failure=args.reward_failure_weight,
    )
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
    execution_audit = []
    validation_history = []
    best_validation = None
    static_validation = []
    if args.validation_seeds:
        static_validation = [evaluate_static_policy(seed, args.steps) for seed in args.validation_seeds]
        static_validation_sla = sum(row["sla_violation_rate"] for row in static_validation) / len(static_validation)
        print(f"validation static reference: sla={static_validation_sla:.4f}")
    else:
        static_validation_sla = None

    for episode in range(args.episodes):
        upper.train()
        lower.train()
        upper_records, lower_records = [], []

        def upper_policy(obs):
            state = torch.tensor(obs.state, dtype=torch.float32, device=device).unsqueeze(0)
            decision_dist, migration_dist, _ = upper_action_distributions(
                upper, state, obs.action_mask, device
            )
            decision = decision_dist.sample()
            log_prob = decision_dist.log_prob(decision)
            if int(decision.item()) == 0:
                planning_action = 0
            else:
                migration_action = migration_dist.sample()
                log_prob = log_prob + migration_dist.log_prob(migration_action)
                planning_action = migration_action_to_planning_action(int(migration_action.item()))
            upper_records.append({"log_prob": log_prob.squeeze(), "value": upper.value(state).squeeze(), "reward": 0.0})
            return planning_action

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
        raw_reward_totals, reward_totals, lower_trajectories = assign_event_rewards(
            result["hac_events"],
            result["service_sla_history"],
            upper_records,
            lower_records,
            reward_weights,
        )
        upper_loss = update_policy(upper, upper_opt, [upper_records], args.gamma)
        lower_loss = update_policy(lower, lower_opt, lower_trajectories, args.gamma)
        execution_attempts = result["upper_immediate_count"] + result["pending_executed"]
        upper_action_count = max(1, len(upper_records))
        execution_events = [event for event in result["hac_events"] if event.execution is not None]
        stage_counts = Counter(event.execution.failure_stage for event in execution_events)
        diagnostics = execution_diagnostics(execution_events)
        execution_raw_totals = {name: 0.0 for name in REWARD_COMPONENTS}
        execution_weighted_totals = {name: 0.0 for name in REWARD_COMPONENTS}
        for event in execution_events:
            _, raw_components, weighted_components = event_reward(event, reward_weights)
            for name in REWARD_COMPONENTS:
                execution_raw_totals[name] += raw_components[name]
                execution_weighted_totals[name] += weighted_components[name]
        execution_event_count = max(1, len(execution_events))
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
            "upper_keep_count": result["upper_keep_count"],
            "upper_immediate_count": result["upper_immediate_count"],
            "upper_delayed_count": result["upper_delayed_count"],
            "scope_link_only_count": result["scope_link_only_count"],
            "scope_partial_count": result["scope_partial_count"],
            "scope_full_count": result["scope_full_count"],
            "pending_created": result["pending_created"],
            "pending_executed": result["pending_executed"],
            "pending_cancelled": result["pending_cancelled"],
            "lower_success": result["lower_success"],
            "lower_failure": result["lower_failure"],
            "routing_success": result["routing_success"],
            "routing_failure": result["routing_failure"],
            "rollback_count": result["rollback_count"],
            "keep_rate": result["upper_keep_count"] / upper_action_count,
            "execution_success_rate": result["metrics"]["migrations"] / max(1, execution_attempts),
            "rollback_rate": result["rollback_count"] / max(1, execution_attempts),
            **diagnostics,
            **{f"execution_stage_{stage}": count for stage, count in sorted(stage_counts.items())},
            **{f"reward_{name}": value for name, value in reward_totals.items()},
            **{f"reward_raw_{name}": value for name, value in raw_reward_totals.items()},
        }
        for name, value in reward_totals.items():
            row[f"reward_{name}_per_upper"] = value / upper_action_count
        for name, value in raw_reward_totals.items():
            row[f"reward_raw_{name}_per_upper"] = value / upper_action_count
        for name in REWARD_COMPONENTS:
            row[f"reward_{name}_per_execution"] = execution_weighted_totals[name] / execution_event_count
            row[f"reward_raw_{name}_per_execution"] = execution_raw_totals[name] / execution_event_count
        history.append(row)
        for event in execution_events:
            execution_audit.append({
                "episode": episode,
                "seed": seed,
                "time_step": event.time_step,
                "service_id": event.service_id,
                "scope": event.decision.scope,
                "event_type": event.event_type,
                "failure_stage": event.execution.failure_stage,
                "migrated": event.execution.outcome.migrated,
                "accepted": event.execution.outcome.accepted,
                "target_vnodes": len(event.execution.plan.target_v_nodes) if event.execution.plan else 0,
                "candidate_counts": event.execution.candidate_counts,
                "released_cpu_ratio": event.released_cpu_ratio,
                "released_bandwidth_ratio": event.released_bandwidth_ratio,
                "pre_risk": event.execution.outcome.pre_risk,
                "post_risk": event.execution.outcome.post_risk,
                "pre_node_risk": event.execution.outcome.pre_node_risk,
                "post_node_risk": event.execution.outcome.post_node_risk,
                "pre_link_risk": event.execution.outcome.pre_link_risk,
                "post_link_risk": event.execution.outcome.post_link_risk,
                "node_risk_reduction": event.execution.outcome.node_risk_reduction,
                "link_risk_reduction": event.execution.outcome.link_risk_reduction,
            })
        print(
            f"episode={episode:03d} upper_loss={upper_loss:.4f} lower_loss={lower_loss:.4f} "
            f"sla={row['sla_violation_rate']:.4f} migrations={row['migrations']} reward={row['reward_total']:.3f}"
        )
        if args.diagnostic:
            print(
                f"  upper={row['upper_actions']} keep={row['upper_keep_count']} "
                f"immediate={row['upper_immediate_count']} delayed={row['upper_delayed_count']} "
                f"pending_exec={row['pending_executed']} pending_cancel={row['pending_cancelled']}"
            )
            print(
                f"  scope: link={row['scope_link_only_count']} partial={row['scope_partial_count']} "
                f"full={row['scope_full_count']} | lower: ok={row['lower_success']} fail={row['lower_failure']} "
                f"routing_fail={row['routing_failure']} rollback={row['rollback_count']}"
            )
            print(f"  execution stages: {dict(sorted(stage_counts.items()))}")
            print(
                f"  accepted={row['accepted_migrations']} avg_risk_delta={row['accepted_avg_risk_reduction']:+.4f} "
                f"avg_cost={row['accepted_avg_realized_cost']:.3f} avg_disruption={row['accepted_avg_disruption']:.3f}"
            )
            print(
                f"  component risk delta: node={row['accepted_avg_node_risk_reduction']:+.4f} "
                f"link={row['accepted_avg_link_risk_reduction']:+.4f}"
            )
            print(
                f"  scope accepted rate: link={row['accepted_rate_scope_link-only']:.3f} "
                f"partial={row['accepted_rate_scope_partial']:.3f} full={row['accepted_rate_scope_full']:.3f}"
            )
            print(
                f"  risk quality: improved={row['accepted_risk_improved_fraction']:.3f} "
                f"worsened={row['accepted_risk_worsened_fraction']:.3f} "
                f"unchanged={row['accepted_risk_unchanged_fraction']:.3f} "
                f"mean_negative={row['accepted_avg_negative_risk_reduction']:+.4f}"
            )
            print(
                f"  reward weighted: risk={row['reward_risk']:+.3f} cost={row['reward_cost']:+.3f} "
                f"disruption={row['reward_disruption']:+.3f} sla={row['reward_sla']:+.3f} "
                f"future_sla={row['reward_future_sla']:+.3f} failure={row['reward_failure']:+.3f} "
                f"per_upper={row['reward_total_per_upper']:+.3f}"
            )
            print(
                f"  reward raw: risk={row['reward_raw_risk']:+.3f} cost={row['reward_raw_cost']:+.3f} "
                f"disruption={row['reward_raw_disruption']:+.3f} sla={row['reward_raw_sla']:+.3f} "
                f"future_sla={row['reward_raw_future_sla']:+.3f} failure={row['reward_raw_failure']:+.3f}"
            )
            print(
                f"  reward per execution: risk={row['reward_risk_per_execution']:+.3f} "
                f"cost={row['reward_cost_per_execution']:+.3f} sla={row['reward_sla_per_execution']:+.3f} "
                f"failure={row['reward_failure_per_execution']:+.3f}"
            )
        if args.validation_seeds and (episode + 1) % args.validation_interval == 0:
            evaluations = [
                evaluate_argmax_policy(upper, lower, seed, args.steps, args.stgcn_checkpoint, device)
                for seed in args.validation_seeds
            ]
            validation = summarize_validation(evaluations, episode + 1)
            validation["val_static_sla_violation_rate"] = static_validation_sla
            validation["val_sla_improvement_over_static"] = (
                static_validation_sla - validation["val_sla_violation_rate"]
            )
            validation_history.append(validation)
            selected = is_better_checkpoint(validation, best_validation, static_validation_sla)
            validation["risk_feasible"] = bool(
                validation["val_accepted_migrations"] > 0
                and validation["val_accepted_avg_risk_reduction"] > CHECKPOINT_EPSILON
                and validation["val_sla_improvement_over_static"] > CHECKPOINT_EPSILON
            )
            validation["selected_as_best"] = selected
            print(
                f"validation episode={episode + 1:03d} sla={validation['val_sla_violation_rate']:.4f} "
                f"risk={validation['val_accepted_avg_risk_reduction']:+.5f} "
                f"sla_gain={validation['val_sla_improvement_over_static']:+.4f} "
                f"keep={validation['val_keep_rate']:.3f} "
                f"migrations={validation['val_migrations']:.2f} feasible={validation['risk_feasible']}"
            )
            if selected:
                best_validation = validation
                checkpoint_payload = {
                    "training_episode": episode + 1,
                    "selection": validation,
                    "stgcn_checkpoint": str(args.stgcn_checkpoint),
                    "reward_weights": asdict(reward_weights),
                    "upper_action_encoding": "binary_keep_migrate_plus_conditional_delay_scope",
                }
                torch.save({"state_dict": upper.state_dict(), **checkpoint_payload}, args.output / "best_upper.pt")
                torch.save({"state_dict": lower.state_dict(), **checkpoint_payload}, args.output / "best_lower.pt")
    calibration_evaluations = [
        evaluate_argmax_policy(upper, lower, seed, args.steps, args.stgcn_checkpoint, device)
        for seed in args.calibration_eval_seeds
    ]
    metadata = {
        "stgcn_checkpoint": str(args.stgcn_checkpoint),
        "selector": "TopRiskKSelector",
        "selector_top_k": 1,
        "hac_enforce_risk_reduction": False,
        "system_risk_aggregation": "max(mean_node_risk, mean_link_risk)",
        "hac_component_risk_diagnostics": True,
        "hac_risk_reward": "node_risk_reduction + link_risk_reduction",
        "upper_state_dim": UPPER_STATE_DIM,
        "upper_action_dim": UPPER_ACTION_DIM,
        "upper_planning_action_dim": UPPER_PLANNING_ACTION_DIM,
        "upper_action_encoding": "binary_keep_migrate_plus_conditional_delay_scope",
        "lower_state_dim": LOWER_STATE_DIM,
        "lower_candidate_dim": LOWER_CANDIDATE_DIM,
        "reward_components": list(REWARD_COMPONENTS),
        "reward_weights": asdict(reward_weights),
        "critic_target": "raw_discounted_return",
        "actor_advantage": "normalized_return_minus_detached_value",
        "calibration_evaluation_policy": "argmax",
        "calibration_evaluation_seeds": args.calibration_eval_seeds,
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
    (args.output / "execution_audit.json").write_text(json.dumps(execution_audit, indent=2), encoding="utf-8")
    (args.output / "calibration_evaluation.json").write_text(
        json.dumps(calibration_evaluations, indent=2), encoding="utf-8"
    )
    (args.output / "validation_history.json").write_text(
        json.dumps(validation_history, indent=2), encoding="utf-8"
    )
    (args.output / "best_metadata.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "accepted_migrations > 0 and accepted_avg_risk_reduction > 0 and "
                    "validation SLA is strictly below matched Static SLA; "
                    "then minimize SLA, total cost, total disruption, migrations"
                ),
                "best_checkpoint_found": best_validation is not None,
                "best_validation": best_validation,
                "static_validation": static_validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
