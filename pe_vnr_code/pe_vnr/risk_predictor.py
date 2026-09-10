from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from .config import PredictorConfig
from .topology import RiskPrediction, TimeVaryingTopology, edge_key


class SimpleSTGCN(nn.Module):
    def __init__(self, node_feature_dim: int, link_feature_dim: int, hidden_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Conv1d(node_feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.link_encoder = nn.Sequential(
            nn.Conv1d(link_feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.node_head = nn.Linear(hidden_dim, 1)
        self.link_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_series: torch.Tensor, link_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_hidden = self.node_encoder(node_series).mean(dim=-1)
        link_hidden = self.link_encoder(link_series).mean(dim=-1)
        node_scores = torch.sigmoid(self.node_head(node_hidden)).squeeze(-1)
        link_scores = torch.sigmoid(self.link_head(link_hidden)).squeeze(-1)
        return node_scores, link_scores


class STRiskPredictor:
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[SimpleSTGCN] = None

    def _node_vector(self, graph: nx.Graph, node_id: int) -> np.ndarray:
        attrs = graph.nodes[node_id]
        cpu = float(attrs.get("cpu", 0.0))
        max_cpu = max(float(attrs.get("max_cpu", max(cpu, 1.0))), 1e-6)
        queue = float(attrs.get("queue", 0.0))
        energy = float(attrs.get("energy", 1.0))
        degree = float(graph.degree[node_id]) / max(1, graph.number_of_nodes() - 1)
        return np.array(
            [
                1.0 - cpu / max_cpu,
                queue,
                1.0 - energy,
                degree,
            ],
            dtype=np.float32,
        )

    def _link_vector(self, graph: nx.Graph, link_id: Tuple[int, int]) -> np.ndarray:
        attrs = graph.edges[link_id]
        bandwidth = float(attrs.get("bandwidth", 0.0))
        max_bandwidth = max(float(attrs.get("max_bandwidth", max(bandwidth, 1.0))), 1e-6)
        delay = float(attrs.get("delay", 0.0))
        loss = float(attrs.get("loss", 0.0))
        visible_time = float(attrs.get("visible_time", 1.0))
        max_delay = max(float(attrs.get("max_delay", 20.0)), 1e-6)
        visibility_horizon = max(float(attrs.get("visibility_horizon", 2.0)), 1e-6)
        return np.array(
            [
                1.0 - bandwidth / max_bandwidth,
                min(1.0, delay / max_delay),
                loss,
                1.0 - min(1.0, visible_time / visibility_horizon),
            ],
            dtype=np.float32,
        )

    def _collect_series(self, topology: TimeVaryingTopology) -> Tuple[List[int], List[Tuple[int, int]], np.ndarray, np.ndarray]:
        history = topology.window(self.config.history_window)
        latest_graph = history[-1]
        nodes = list(latest_graph.nodes)
        links = [edge_key(u, v) for u, v in latest_graph.edges]

        node_series = []
        link_series = []
        for graph in history:
            node_series.append([self._node_vector(graph, node_id) for node_id in nodes])
            link_series.append([self._link_vector(graph, link_id) for link_id in links])
        return nodes, links, np.array(node_series), np.array(link_series)

    def _heuristic_predict(self, topology: TimeVaryingTopology) -> RiskPrediction:
        nodes, links, node_series, link_series = self._collect_series(topology)
        node_now = node_series[-1]
        link_now = link_series[-1]
        node_var = node_series.var(axis=0)
        link_var = link_series.var(axis=0)

        node_risk = {}
        for index, node_id in enumerate(nodes):
            pressure = 0.60 * node_now[index, 0] + 0.15 * node_now[index, 1] + 0.15 * node_now[index, 2]
            volatility = 0.10 * node_var[index].mean()
            node_risk[node_id] = float(min(1.0, pressure + volatility))

        link_risk = {}
        for index, link_id in enumerate(links):
            pressure = 0.45 * link_now[index, 0] + 0.20 * link_now[index, 1] + 0.15 * link_now[index, 2] + 0.15 * link_now[index, 3]
            volatility = 0.05 * link_var[index].mean()
            link_risk[link_id] = float(min(1.0, pressure + volatility))

        future_node_risk = [dict(node_risk) for _ in range(self.config.future_horizon)]
        future_link_risk = [dict(link_risk) for _ in range(self.config.future_horizon)]
        return RiskPrediction(node_risk, link_risk, future_node_risk, future_link_risk)

    def _learned_predict(self, topology: TimeVaryingTopology) -> RiskPrediction:
        nodes, links, node_series, link_series = self._collect_series(topology)
        node_tensor = torch.tensor(node_series, dtype=torch.float32, device=self.device).permute(1, 2, 0)
        link_tensor = torch.tensor(link_series, dtype=torch.float32, device=self.device).permute(1, 2, 0)
        if self.model is None:
            self.model = SimpleSTGCN(node_tensor.shape[1], link_tensor.shape[1], self.config.hidden_dim).to(self.device)
        with torch.no_grad():
            node_scores, link_scores = self.model(node_tensor, link_tensor)
        node_risk = {node_id: float(score) for node_id, score in zip(nodes, node_scores.cpu().tolist())}
        link_risk = {link_id: float(score) for link_id, score in zip(links, link_scores.cpu().tolist())}
        future_node_risk = [dict(node_risk) for _ in range(self.config.future_horizon)]
        future_link_risk = [dict(link_risk) for _ in range(self.config.future_horizon)]
        return RiskPrediction(node_risk, link_risk, future_node_risk, future_link_risk)

    def predict(self, topology: TimeVaryingTopology) -> RiskPrediction:
        if len(topology.history) < 2 or not self.config.use_learned_model:
            return self._heuristic_predict(topology)
        try:
            return self._learned_predict(topology)
        except Exception:
            return self._heuristic_predict(topology)
