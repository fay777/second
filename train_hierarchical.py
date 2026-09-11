"""Episodic Hierarchical Actor-Critic trainer for planning and execution MDPs.

The risk predictor is intentionally heuristic here.  Replace it with a
validated ST-GCN checkpoint before reporting learned-policy results in a paper.

Returns are computed over complete episodes with a learned value baseline.  It
is therefore described as HAC rather than claiming a strict n-step A2C update.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from pe_vnr.mdp.actor_critic import ExecutionActorCritic, PlanningActorCritic
from pe_vnr.components import ComponentConfig, build_environment


def build_training_environment(
    seed: int,
    predictor: str,
    predictor_checkpoint: Path = None,
    scenario: str = "toy",
    physical_nodes: int = 6,
    virtual_nodes: int = 3,
    history_window: int = 4,
    future_horizon: int = 3,
):
    return build_environment(
        ComponentConfig(
            mode="heuristic-proactive",
            predictor=predictor,
            predictor_checkpoint=predictor_checkpoint,
            seed=seed,
            scenario=scenario,
            physical_nodes=physical_nodes,
            virtual_nodes=virtual_nodes,
            history_window=history_window,
            future_horizon=future_horizon,
        )
    )


def select_planning_action(model, observation, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(observation.state, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(observation.action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    logits = model.act(state).masked_fill(~mask, -1e9)
    distribution = Categorical(logits=logits)
    action = distribution.sample()
    return int(action.item()), distribution.log_prob(action).squeeze(0), model.value(state).squeeze()


def select_execution_action(model, observation, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(observation.state, dtype=torch.float32, device=device).unsqueeze(0)
    candidates = torch.as_tensor(observation.candidate_features, dtype=torch.float32, device=device).unsqueeze(0)
    distribution = Categorical(logits=model.act(state, candidates))
    action = distribution.sample()
    return int(action.item()), distribution.log_prob(action).squeeze(0), model.value(state).squeeze()


def update(model, optimizer, log_probs: List[torch.Tensor], values: List[torch.Tensor], rewards: List[float], gamma: float) -> float:
    if not rewards:
        return 0.0
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=values[0].device)
    values_tensor = torch.stack(values)
    advantages = returns - values_tensor.detach()
    policy_loss = -(torch.stack(log_probs) * advantages).mean()
    value_loss = torch.nn.functional.mse_loss(values_tensor, returns)
    loss = policy_loss + 0.5 * value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def train(
    episodes: int,
    gamma: float,
    learning_rate: float,
    seed: int,
    device: torch.device,
    predictor: str,
    predictor_checkpoint: Path,
    scenario: str,
    physical_nodes: int,
    virtual_nodes: int,
    history_window: int,
    future_horizon: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe = build_training_environment(seed, predictor, predictor_checkpoint, scenario, physical_nodes, virtual_nodes, history_window, future_horizon)
    planning_observation = probe.reset()
    planning_model = PlanningActorCritic(planning_observation.state.size, planning_observation.action_mask.size).to(device)

    # A full plan guarantees at least one target and is used only to infer lower-policy dimensions.
    execution_probe = probe.planning_step(probe.planning_env.encode_action(1, 0, 2))
    if execution_probe.stage != "execution":
        raise RuntimeError("Unable to initialize the execution MDP for training.")
    execution_model = ExecutionActorCritic(
        execution_probe.observation.state.size,
        execution_probe.observation.candidate_features.shape[1],
    ).to(device)
    planning_optimizer = torch.optim.Adam(planning_model.parameters(), lr=learning_rate)
    execution_optimizer = torch.optim.Adam(execution_model.parameters(), lr=learning_rate)

    history = []
    for episode in range(episodes):
        environment = build_training_environment(
            seed + episode,
            predictor,
            predictor_checkpoint,
            scenario,
            physical_nodes,
            virtual_nodes,
            history_window,
            future_horizon,
        )
        observation = environment.reset()
        planning_log_probs: List[torch.Tensor] = []
        planning_values: List[torch.Tensor] = []
        planning_rewards: List[float] = []
        execution_log_probs: List[torch.Tensor] = []
        execution_values: List[torch.Tensor] = []
        execution_rewards: List[float] = []
        episode_reward = 0.0
        active_plan_index = None

        while True:
            if environment.training_stage == "planning":
                action, log_prob, value = select_planning_action(planning_model, observation, device)
                transition = environment.planning_step(action)
                planning_log_probs.append(log_prob)
                planning_values.append(value)
                planning_rewards.append(transition.reward)
                active_plan_index = len(planning_rewards) - 1
            else:
                action, log_prob, value = select_execution_action(execution_model, observation, device)
                transition = environment.execution_step(action)
                execution_log_probs.append(log_prob)
                execution_values.append(value)
                execution_rewards.append(transition.reward)
                if active_plan_index is not None:
                    planning_rewards[active_plan_index] += transition.reward
            episode_reward += transition.reward
            if transition.terminated:
                break
            observation = transition.observation

        planning_loss = update(planning_model, planning_optimizer, planning_log_probs, planning_values, planning_rewards, gamma)
        execution_loss = update(execution_model, execution_optimizer, execution_log_probs, execution_values, execution_rewards, gamma)
        history.append(episode_reward)
        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"episode={episode + 1:04d} reward={episode_reward:.3f} "
                f"planning_loss={planning_loss:.4f} execution_loss={execution_loss:.4f}"
            )
    return planning_model, execution_model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical planning/execution Actor-Critic policies.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--predictor", choices=("heuristic", "stgcn"), default="heuristic")
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--scenario", choices=("toy", "mobility", "congestion", "outage", "compound"), default="toy")
    parser.add_argument("--physical-nodes", type=int, default=6)
    parser.add_argument("--virtual-nodes", type=int, default=3)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--future-horizon", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.predictor == "stgcn" and args.predictor_checkpoint is None:
        raise ValueError("--predictor-checkpoint is required when --predictor stgcn")
    planning_model, execution_model, rewards = train(
        args.episodes, args.gamma, args.learning_rate, args.seed, device,
        args.predictor, args.predictor_checkpoint, args.scenario,
        args.physical_nodes, args.virtual_nodes, args.history_window, args.future_horizon,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"predictor": args.predictor, "scenario": args.scenario, "physical_nodes": args.physical_nodes, "virtual_nodes": args.virtual_nodes, "history_window": args.history_window, "future_horizon": args.future_horizon}
    torch.save({"state_dict": planning_model.state_dict(), "component": "upper-planning-ac", **metadata}, args.output / "planning_actor_critic.pt")
    torch.save({"state_dict": execution_model.state_dict(), "component": "lower-execution-ac", **metadata}, args.output / "execution_actor_critic.pt")
    np.savetxt(args.output / "training_rewards.csv", np.asarray(rewards), delimiter=",")
    print(f"Saved checkpoints and rewards to {args.output}")


if __name__ == "__main__":
    main()
