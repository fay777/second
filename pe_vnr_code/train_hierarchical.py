"""Minimal masked A2C trainer for the planning and execution MDPs.

The risk predictor is intentionally heuristic here.  Replace it with a
validated ST-GCN checkpoint before reporting learned-policy results in a paper.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from pe_vnr.mdp.actor_critic import ExecutionActorCritic, PlanningActorCritic
from run_baselines import build_environment


def select_action(model, observation, device: torch.device) -> Tuple[int, torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(observation.state, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(observation.action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    logits = model.act(state).masked_fill(~mask, -1e9)
    distribution = Categorical(logits=logits)
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


def train(episodes: int, gamma: float, learning_rate: float, seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe = build_environment(seed, "proactive_heuristic")
    planning_observation = probe.reset()
    planning_model = PlanningActorCritic(planning_observation.state.size, planning_observation.action_mask.size).to(device)

    # A partial plan always has at least one target virtual node in this environment.
    execution_probe = probe.planning_step(probe.planning_env.encode_action(1, 0, 1))
    if execution_probe.stage != "execution":
        raise RuntimeError("Unable to initialize the execution MDP for training.")
    execution_model = ExecutionActorCritic(
        execution_probe.observation.state.size,
        execution_probe.observation.action_mask.size,
    ).to(device)
    planning_optimizer = torch.optim.Adam(planning_model.parameters(), lr=learning_rate)
    execution_optimizer = torch.optim.Adam(execution_model.parameters(), lr=learning_rate)

    history = []
    for episode in range(episodes):
        environment = build_environment(seed + episode, "proactive_heuristic")
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
                action, log_prob, value = select_action(planning_model, observation, device)
                transition = environment.planning_step(action)
                planning_log_probs.append(log_prob)
                planning_values.append(value)
                planning_rewards.append(transition.reward)
                active_plan_index = len(planning_rewards) - 1
            else:
                action, log_prob, value = select_action(execution_model, observation, device)
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
    parser = argparse.ArgumentParser(description="Train masked planning/execution Actor-Critic policies.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planning_model, execution_model, rewards = train(
        args.episodes,
        args.gamma,
        args.learning_rate,
        args.seed,
        device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(planning_model.state_dict(), args.output / "planning_actor_critic.pt")
    torch.save(execution_model.state_dict(), args.output / "execution_actor_critic.pt")
    np.savetxt(args.output / "training_rewards.csv", np.asarray(rewards), delimiter=",")
    print(f"Saved checkpoints and rewards to {args.output}")


if __name__ == "__main__":
    main()
