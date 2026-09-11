from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from .config import PredictorConfig
from .topology import RiskPrediction, TimeVaryingTopology, edge_key


class SpatialGraphConv(nn.Module):
    """Normalized adjacency aggregation used by the spatial ST-GCN block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_hidden: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.projection(normalized_adjacency @ node_hidden))


class STGCNRiskNet(nn.Module):
    """Per-snapshot spatial GCN followed by temporal convolution and multi-step heads."""

    def __init__(self, node_feature_dim: int, link_feature_dim: int, hidden_dim: int, future_horizon: int):
        super().__init__()
        self.future_horizon = future_horizon
        self.node_input = nn.Linear(node_feature_dim, hidden_dim)
        self.link_input = nn.Linear(link_feature_dim, hidden_dim)
        self.node_temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.node_spatial = SpatialGraphConv(hidden_dim)
        self.link_temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.link_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.node_head = nn.Linear(hidden_dim, future_horizon)
        self.link_head = nn.Linear(hidden_dim, future_horizon)

    def forward(
        self,
        node_series: torch.Tensor,
        link_series: torch.Tensor,
        normalized_adjacency_series: torch.Tensor,
        link_endpoints: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Each historical snapshot has its own adjacency matrix before temporal aggregation.
        node_steps = self.node_input(node_series.permute(2, 0, 1))
        spatial_steps = torch.stack(
            [self.node_spatial(node_steps[time_index], normalized_adjacency_series[time_index]) for time_index in range(node_steps.shape[0])]
        )
        node_hidden = self.node_temporal(spatial_steps.permute(1, 2, 0)).mean(dim=-1)
        link_steps = self.link_input(link_series.permute(2, 0, 1))
        endpoint_steps = 0.5 * (
            spatial_steps[:, link_endpoints[:, 0]] + spatial_steps[:, link_endpoints[:, 1]]
        )
        link_steps = torch.relu(self.link_fusion(torch.cat([link_steps, endpoint_steps], dim=-1)))
        link_hidden = self.link_temporal(link_steps.permute(1, 2, 0)).mean(dim=-1)
        node_scores = torch.sigmoid(self.node_head(node_hidden))
        link_scores = torch.sigmoid(self.link_head(link_hidden))
        return node_scores, link_scores


class STRiskPredictor:
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[STGCNRiskNet] = None
        self._checkpoint_loaded = False

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

    @staticmethod
    def _graph_tensors(
        nodes: List[int],
        links: List[Tuple[int, int]],
        graphs: List[nx.Graph],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        index = {node_id: position for position, node_id in enumerate(nodes)}
        adjacencies = []
        for graph in graphs:
            adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
            for u, v, attrs in graph.edges(data=True):
                if u in index and v in index:
                    # Failed, depleted, or invisible links do not propagate messages.
                    if (
                        float(attrs.get("fault", 0.0)) >= 1.0
                        or float(attrs.get("bandwidth", 0.0)) <= 0.0
                        or float(attrs.get("visible_time", 1.0)) <= 0.0
                    ):
                        continue
                    adjacency[index[u], index[v]] = 1.0
                    adjacency[index[v], index[u]] = 1.0
            adjacency += np.eye(len(nodes), dtype=np.float32)
            degrees = np.maximum(adjacency.sum(axis=1), 1e-6)
            adjacencies.append(adjacency / np.sqrt(degrees[:, None] * degrees[None, :]))
        endpoints = np.array([[index[u], index[v]] for u, v in links], dtype=np.int64)
        return torch.tensor(np.stack(adjacencies), device=device), torch.tensor(endpoints, device=device)

    def ground_truth_risk(self, graph: nx.Graph) -> RiskPrediction:
        """Risk labels derived directly from the observed future snapshot and fault events."""
        node_risk = {}
        for node_id in graph.nodes:
            vector = self._node_vector(graph, node_id)
            node_risk[node_id] = float(min(1.0, 0.40 * vector[0] + 0.25 * vector[1] + 0.20 * vector[2] + 0.15 * float(graph.nodes[node_id].get("fault", 0.0))))
        link_risk = {}
        for u, v in graph.edges:
            link_id = edge_key(u, v)
            vector = self._link_vector(graph, link_id)
            link_risk[link_id] = float(min(1.0, 0.30 * vector[0] + 0.20 * vector[1] + 0.15 * vector[2] + 0.15 * vector[3] + 0.20 * float(graph.edges[link_id].get("fault", 0.0))))
        return RiskPrediction(node_risk, link_risk, [dict(node_risk)], [dict(link_risk)])

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
        adjacency, endpoints = self._graph_tensors(nodes, links, topology.window(self.config.history_window), self.device)
        if self.model is None:
            self.model = STGCNRiskNet(
                node_tensor.shape[1],
                link_tensor.shape[1],
                self.config.hidden_dim,
                self.config.future_horizon,
            ).to(self.device)
        if self.config.checkpoint_path and not self._checkpoint_loaded:
            try:
                checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            metadata = checkpoint if isinstance(checkpoint, dict) else {}
            expected = {
                "history_window": self.config.history_window,
                "future_horizon": self.config.future_horizon,
                "hidden_dim": self.config.hidden_dim,
                "node_feature_dim": node_tensor.shape[1],
                "link_feature_dim": link_tensor.shape[1],
            }
            missing = [name for name in expected if name not in metadata]
            mismatched = [name for name, value in expected.items() if name in metadata and metadata[name] != value]
            if missing or mismatched:
                raise ValueError(
                    "ST-GCN checkpoint metadata is incompatible with the active predictor: "
                    f"missing={missing}, mismatched={mismatched}."
                )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._checkpoint_loaded = True
        if self.config.require_checkpoint and not self._checkpoint_loaded:
            raise RuntimeError("ST-GCN mode requires a trained predictor checkpoint.")
        with torch.no_grad():
            node_scores, link_scores = self.model(node_tensor, link_tensor, adjacency, endpoints)
        future_node_risk = [
            {node_id: float(node_scores[index, horizon].item()) for index, node_id in enumerate(nodes)}
            for horizon in range(self.config.future_horizon)
        ]
        future_link_risk = [
            {link_id: float(link_scores[index, horizon].item()) for index, link_id in enumerate(links)}
            for horizon in range(self.config.future_horizon)
        ]
        node_risk = future_node_risk[0]
        link_risk = future_link_risk[0]
        return RiskPrediction(node_risk, link_risk, future_node_risk, future_link_risk)

    def predict(self, topology: TimeVaryingTopology) -> RiskPrediction:
        if len(topology.history) < 2 or not self.config.use_learned_model:
            return self._heuristic_predict(topology)
        try:
            return self._learned_predict(topology)
        except Exception:
            if self.config.require_checkpoint:
                raise
            return self._heuristic_predict(topology)
