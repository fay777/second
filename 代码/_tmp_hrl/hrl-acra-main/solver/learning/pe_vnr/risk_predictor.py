import copy
from typing import List

import networkx as nx
import numpy as np
import torch

from .net import RiskPrediction, TemporalRiskEncoder


class STRiskPredictor:
    def __init__(self, controller, hidden_dim=64, use_learned_model=False, device="cpu"):
        self.controller = controller
        self.hidden_dim = hidden_dim
        self.use_learned_model = use_learned_model
        self.device = torch.device(device)
        self.model = None

    def _node_resource_ratio(self, p_net, node_id):
        ratios = []
        for attr in self.controller.node_resource_attrs:
            max_name = f"max_{attr.name}"
            curr = float(p_net.nodes[node_id].get(attr.name, 0.0))
            cap = float(p_net.nodes[node_id].get(max_name, max(curr, 1.0)))
            cap = max(cap, 1e-6)
            ratios.append(1.0 - curr / cap)
        if not ratios:
            return 0.0
        return float(np.mean(ratios))

    def _link_resource_ratio(self, p_net, link_id):
        ratios = []
        for attr in self.controller.link_resource_attrs:
            max_name = f"max_{attr.name}"
            curr = float(p_net.links[link_id].get(attr.name, 0.0))
            cap = float(p_net.links[link_id].get(max_name, max(curr, 1.0)))
            cap = max(cap, 1e-6)
            ratios.append(1.0 - curr / cap)
        if not ratios:
            return 0.0
        return float(np.mean(ratios))

    def _build_snapshot_features(self, p_net):
        degree_centrality = nx.degree_centrality(p_net) if len(p_net.nodes) > 1 else {n: 0.0 for n in p_net.nodes}
        node_features = {}
        for node_id in p_net.nodes:
            node_features[node_id] = np.array(
                [
                    self._node_resource_ratio(p_net, node_id),
                    float(degree_centrality.get(node_id, 0.0)),
                ],
                dtype=np.float32,
            )
        link_features = {}
        for link_id in p_net.links:
            link_features[link_id] = np.array(
                [
                    self._link_resource_ratio(p_net, link_id),
                ],
                dtype=np.float32,
            )
        return node_features, link_features

    def _build_temporal_tensors(self, history_p_nets: List):
        history_node_features = []
        history_link_features = []
        latest_p_net = history_p_nets[-1]
        latest_nodes = list(latest_p_net.nodes)
        latest_links = list(latest_p_net.links)
        for p_net in history_p_nets:
            node_features, link_features = self._build_snapshot_features(p_net)
            history_node_features.append([node_features[node_id] for node_id in latest_nodes])
            history_link_features.append([link_features[link_id] for link_id in latest_links])
        node_tensor = torch.tensor(np.array(history_node_features), dtype=torch.float32, device=self.device).transpose(0, 1)
        link_tensor = torch.tensor(np.array(history_link_features), dtype=torch.float32, device=self.device).transpose(0, 1)
        return latest_nodes, latest_links, node_tensor, link_tensor

    def _heuristic_predict(self, history_p_nets: List):
        latest_p_net = history_p_nets[-1]
        latest_nodes = list(latest_p_net.nodes)
        latest_links = list(latest_p_net.links)
        history_node_ratios = {node_id: [] for node_id in latest_nodes}
        history_link_ratios = {link_id: [] for link_id in latest_links}

        for p_net in history_p_nets:
            for node_id in latest_nodes:
                history_node_ratios[node_id].append(self._node_resource_ratio(p_net, node_id))
            for link_id in latest_links:
                history_link_ratios[link_id].append(self._link_resource_ratio(p_net, link_id))

        node_risk = {}
        for node_id, ratios in history_node_ratios.items():
            mean_ratio = float(np.mean(ratios))
            volatility = float(np.std(ratios))
            node_risk[node_id] = min(1.0, 0.7 * mean_ratio + 0.3 * volatility)

        link_risk = {}
        for link_id, ratios in history_link_ratios.items():
            mean_ratio = float(np.mean(ratios))
            volatility = float(np.std(ratios))
            link_risk[link_id] = min(1.0, 0.7 * mean_ratio + 0.3 * volatility)

        return RiskPrediction(
            node_risk=node_risk,
            link_risk=link_risk,
            mean_node_risk=float(np.mean(list(node_risk.values()))) if node_risk else 0.0,
            mean_link_risk=float(np.mean(list(link_risk.values()))) if link_risk else 0.0,
            max_node_risk=float(np.max(list(node_risk.values()))) if node_risk else 0.0,
            max_link_risk=float(np.max(list(link_risk.values()))) if link_risk else 0.0,
        )

    def predict(self, history_p_nets: List):
        if history_p_nets is None or len(history_p_nets) == 0:
            raise ValueError("history_p_nets is required for proactive reconfiguration.")

        if not self.use_learned_model or len(history_p_nets) < 2:
            return self._heuristic_predict(history_p_nets)

        latest_nodes, latest_links, node_tensor, link_tensor = self._build_temporal_tensors(history_p_nets)
        if self.model is None:
            self.model = TemporalRiskEncoder(node_tensor.shape[-1], hidden_dim=self.hidden_dim).to(self.device)
        with torch.no_grad():
            node_scores, link_scores = self.model(node_tensor, link_tensor)
        node_risk = {node_id: float(score) for node_id, score in zip(latest_nodes, node_scores.cpu().tolist())}
        link_risk = {link_id: float(score) for link_id, score in zip(latest_links, link_scores.cpu().tolist())}
        return RiskPrediction(
            node_risk=node_risk,
            link_risk=link_risk,
            mean_node_risk=float(np.mean(list(node_risk.values()))) if node_risk else 0.0,
            mean_link_risk=float(np.mean(list(link_risk.values()))) if link_risk else 0.0,
            max_node_risk=float(np.max(list(node_risk.values()))) if node_risk else 0.0,
            max_link_risk=float(np.max(list(link_risk.values()))) if link_risk else 0.0,
        )
