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
    """Upper policy: whether, when, and how broadly to reconfigure."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = MLP(state_dim, action_dim, hidden_dim)
        self.critic = MLP(state_dim, 1, hidden_dim)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)


class ExecutionActorCritic(nn.Module):
    """Candidate-wise lower policy that scores each feasible physical node."""
    def __init__(self, state_dim: int, candidate_feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actor = MLP(state_dim + candidate_feature_dim, 1, hidden_dim)
        self.critic = MLP(state_dim, 1, hidden_dim)

    def act(self, state: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """Return one logit per candidate; candidate count can vary per decision."""
        if candidate_features.dim() == 2:
            candidate_features = candidate_features.unsqueeze(0)
        expanded_state = state.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        logits = self.actor(torch.cat([expanded_state, candidate_features], dim=-1)).squeeze(-1)
        return logits

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)
