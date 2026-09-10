from dataclasses import dataclass

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalRiskEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.node_head = MLP(hidden_dim, hidden_dim, 1, num_layers=2)
        self.link_head = MLP(hidden_dim, hidden_dim, 1, num_layers=2)

    def forward(self, node_series, link_series):
        node_hidden, _ = self.temporal(node_series)
        link_hidden, _ = self.temporal(link_series)
        node_score = torch.sigmoid(self.node_head(node_hidden[:, -1, :])).squeeze(-1)
        link_score = torch.sigmoid(self.link_head(link_hidden[:, -1, :])).squeeze(-1)
        return node_score, link_score


@dataclass
class RiskPrediction:
    node_risk: dict
    link_risk: dict
    mean_node_risk: float
    mean_link_risk: float
    max_node_risk: float
    max_link_risk: float
