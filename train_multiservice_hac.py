"""Train Upper/Lower HAC on the frozen ST-GCN + Top-1 multi-VNR environment."""

import argparse
import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
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
UPPER_PLANNING_ACTION_DIM = 24  # Existing PlanningEnv encoding.
UPPER_ACTION_DIM = 13  # One canonical KEEP action + 4 delays x 3 migration scopes.
UPPER_MIGRATION_OFFSET = UPPER_PLANNING_ACTION_DIM - (UPPER_ACTION_DIM - 1)
FUTURE_SLA_HORIZON = 3
REWARD_COMPONENTS = ("risk", "cost", "disruption", "sla", "future_sla", "failure")


@dataclass(frozen=True)
class RewardWeights:
    """Explicit reward weights; defaults reproduce the prior reward definition."""

    risk: float = 2.0
    cost: float = 1.0
    disruption: float = 1.0
    sla: float = 1.0
    future_sla: float = 1.0
    failure: float = 1.0


def compact_upper_action_mask(planning_mask, device) -> torch.Tensor:
    """Remove redundant KEEP encodings so migration actions have no cardinality prior."""
    compact_mask = [bool(planning_mask[0]), *(bool(value) for value in planning_mask[UPPER_MIGRATION_OFFSET:])]
    return torch.tensor(compact_mask, dtype=torch.bool, device=device).unsqueeze(0)


def policy_action_to_planning_action(action: int) -> int:
    """Map canonical policy action 0=KEEP, 1..12=migration to PlanningEnv encoding."""
    if not 0 <= action < UPPER_ACTION_DIM:
        raise ValueError(f"Upper policy action is outside canonical action space: {action}")
    return 0 if action == 0 else action + UPPER_MIGRATION_OFFSET - 1


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
        mask = compact_upper_action_mask(obs.action_mask, device)
        logits = upper.act(state).masked_fill(~mask, -1e9)
        probabilities = torch.softmax(logits, dim=-1)
        keep_probabilities.append(float(probabilities[0, 0].item()))
        migration_logits = logits[0, 1:]
        if torch.any(mask[0, 1:]):
            migration_margins.append(float((migration_logits.max() - logits[0, 0]).item()))
        return policy_action_to_planning_action(int(logits.argmax(dim=-1).item()))

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
        "upper_keep_count": result["upper_keep_count"],
        "upper_actions": sum(event.event_type == "upper_decision" for event in result["hac_events"]),
        "keep_rate": result["upper_keep_count"] / max(1, sum(event.event_type == "upper_decision" for event in result["hac_events"])),
        "mean_keep_probability": sum(keep_probabilities) / max(1, len(keep_probabilities)),
        "mean_migration_vs_keep_logit_margin": sum(migration_margins) / max(1, len(migration_margins)),
        "no_candidate_host_rate": stage_counts["no_candidate_host"] / max(1, len(execution_events)),
        "execution_stages": dict(sorted(stage_counts.items())),
        **diagnostics,
    }


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostic", action="store_true", help="Print HAC action and reward-component diagnostics per episode.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/multiservice_hac"))
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("episodes and steps must be positive.")
    if not args.stgcn_checkpoint.is_file():
        raise FileNotFoundError(f"Missing ST-GCN checkpoint: {args.stgcn_checkpoint}")
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

    for episode in range(args.episodes):
        upper_records, lower_records = [], []

        def upper_policy(obs):
            state = torch.tensor(obs.state, dtype=torch.float32, device=device).unsqueeze(0)
            mask = compact_upper_action_mask(obs.action_mask, device)
            dist = Categorical(logits=upper.act(state).masked_fill(~mask, -1e9))
            action = dist.sample()
            upper_records.append({"log_prob": dist.log_prob(action).squeeze(), "value": upper.value(state).squeeze(), "reward": 0.0})
            return policy_action_to_planning_action(int(action.item()))

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
        "upper_action_encoding": "canonical_keep_plus_delay_scope_migration",
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


if __name__ == "__main__":
    main()
