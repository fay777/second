import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlanningActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = MLP(state_dim, action_dim, hidden_dim)
        self.critic = MLP(state_dim, 1, hidden_dim)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)


class ExecutionActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = MLP(state_dim, action_dim, hidden_dim)
        self.critic = MLP(state_dim, 1, hidden_dim)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)
