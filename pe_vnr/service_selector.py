"""Risk-aware scheduling of running services before reconfiguration."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx

from .topology import RiskPrediction, RunningService, edge_key


@dataclass(frozen=True)
class ServiceSelectionRecord:
    service_id: str
    peak_risk: float
    expected_risk: float
    priority: float
    score: float
    rank: Optional[int]
    selected: bool
    has_pending_plan: bool


@dataclass(frozen=True)
class ServiceSelection:
    selected_service_ids: List[str]
    records: List[ServiceSelectionRecord]


class RiskAwareServiceSelector:
    """Rank services by predicted exposure without adding another RL layer."""

    def __init__(self, risk_threshold: float = 0.50, peak_weight: float = 0.70, top_k: Optional[int] = None):
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in [0, 1]")
        if not 0.0 <= peak_weight <= 1.0:
            raise ValueError("peak_weight must be in [0, 1]")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None for all candidates")
        self.risk_threshold = risk_threshold
        self.peak_weight = peak_weight
        self.top_k = top_k

    @staticmethod
    def _mapped_resources(service: RunningService) -> Tuple[List[int], List[Tuple[int, int]]]:
        nodes = list(service.deployment.node_mapping.values())
        links = [edge_key(*link_id) for path in service.deployment.link_mapping.values() for link_id in path]
        return nodes, links

    @staticmethod
    def _max_risk(resources: Iterable, maps: List[Dict]) -> float:
        values = [risk_map.get(resource, 0.0) for risk_map in maps for resource in resources]
        return max(values, default=0.0)

    @staticmethod
    def _mean_map_risk(resources: Iterable, risk_map: Dict) -> float:
        values = [risk_map.get(resource, 0.0) for resource in resources]
        return sum(values) / len(values) if values else 0.0

    def _exposure(self, service: RunningService, prediction: RiskPrediction) -> Tuple[float, float]:
        nodes, links = self._mapped_resources(service)
        node_steps = prediction.future_node_risk or [prediction.node_risk]
        link_steps = prediction.future_link_risk or [prediction.link_risk]
        peak = max(self._max_risk(nodes, node_steps), self._max_risk(links, link_steps))
        expected_nodes, expected_links = prediction.expected_risk_maps()
        expected_values = [
            self._mean_map_risk(nodes, expected_nodes),
            self._mean_map_risk(links, expected_links),
        ]
        expected = sum(expected_values) / len(expected_values)
        return peak, expected

    def select(
        self,
        services: Iterable[RunningService],
        prediction: RiskPrediction,
        graph: nx.Graph,
        pending_plans: Mapping[str, object],
    ) -> ServiceSelection:
        del graph  # The graph is reserved for future cost-aware selector variants.
        candidates = []
        records: List[ServiceSelectionRecord] = []
        for service in services:
            peak, expected = self._exposure(service, prediction)
            pending = service.service_id in pending_plans
            score = float(service.priority) * (self.peak_weight * peak + (1.0 - self.peak_weight) * expected)
            if not pending and peak >= self.risk_threshold:
                candidates.append((service.service_id, peak, expected, float(service.priority), score))
            else:
                records.append(ServiceSelectionRecord(service.service_id, peak, expected, float(service.priority), score, None, False, pending))

        candidates.sort(key=lambda item: (-item[4], item[0]))
        selected_candidates = candidates if self.top_k is None else candidates[: self.top_k]
        selected_ids = [item[0] for item in selected_candidates]
        selected_set = set(selected_ids)
        ranks = {item[0]: index + 1 for index, item in enumerate(candidates)}
        records.extend(
            ServiceSelectionRecord(
                service_id, peak, expected, priority, score, ranks[service_id], service_id in selected_set, False
            )
            for service_id, peak, expected, priority, score in candidates
        )
        records.sort(key=lambda record: record.service_id)
        return ServiceSelection(selected_ids, records)


class AllRiskySelector(RiskAwareServiceSelector):
    def __init__(self, risk_threshold: float = 0.50, peak_weight: float = 0.70):
        super().__init__(risk_threshold=risk_threshold, peak_weight=peak_weight, top_k=None)


class TopRiskKSelector(RiskAwareServiceSelector):
    def __init__(self, top_k: int = 1, risk_threshold: float = 0.50, peak_weight: float = 0.70):
        super().__init__(risk_threshold=risk_threshold, peak_weight=peak_weight, top_k=top_k)
