"""Non-graph prediction baselines used to isolate ST-GCN spatial benefits."""

import torch
import torch.nn as nn


class TemporalOnlyRiskNet(nn.Module):
    """Independent temporal convolutions; deliberately receives no adjacency."""

    def __init__(self, node_feature_dim: int, link_feature_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Conv1d(node_feature_dim, hidden_dim, 3, padding=1), nn.ReLU(), nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU())
        self.link_encoder = nn.Sequential(nn.Conv1d(link_feature_dim, hidden_dim, 3, padding=1), nn.ReLU(), nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU())
        self.node_head = nn.Linear(hidden_dim, horizon)
        self.link_head = nn.Linear(hidden_dim, horizon)

    def forward(self, node_series, link_series, _adjacency=None, _endpoints=None):
        node_hidden = self.node_encoder(node_series).mean(dim=-1)
        link_hidden = self.link_encoder(link_series).mean(dim=-1)
        return torch.sigmoid(self.node_head(node_hidden)), torch.sigmoid(self.link_head(link_hidden))


class PersistenceRiskNet(nn.Module):
    """Copies last-step pressure estimates over all prediction horizons."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def forward(self, node_series, link_series, _adjacency=None, _endpoints=None):
        node = (0.40 * node_series[:, 0, -1] + 0.25 * node_series[:, 1, -1] + 0.20 * node_series[:, 2, -1]).clamp(0.0, 1.0)
        link = (0.30 * link_series[:, 0, -1] + 0.20 * link_series[:, 1, -1] + 0.15 * link_series[:, 2, -1] + 0.15 * link_series[:, 3, -1]).clamp(0.0, 1.0)
        return node.unsqueeze(1).repeat(1, self.horizon), link.unsqueeze(1).repeat(1, self.horizon)


class HeuristicTrendRiskNet(PersistenceRiskNet):
    """Pressure heuristic with a bounded observed temporal trend extrapolation."""

    def forward(self, node_series, link_series, _adjacency=None, _endpoints=None):
        node_current, link_current = super().forward(node_series, link_series)
        node_trend = (node_series[:, :3, -1].mean(dim=1) - node_series[:, :3, 0].mean(dim=1)).unsqueeze(1)
        link_trend = (link_series[:, :4, -1].mean(dim=1) - link_series[:, :4, 0].mean(dim=1)).unsqueeze(1)
        steps = torch.arange(1, self.horizon + 1, device=node_series.device, dtype=node_series.dtype).unsqueeze(0)
        return (node_current + 0.10 * node_trend * steps).clamp(0.0, 1.0), (link_current + 0.10 * link_trend * steps).clamp(0.0, 1.0)
