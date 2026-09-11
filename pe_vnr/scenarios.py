"""Configurable dynamic SAGIN scenarios for matched experiments."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx

from .topology import Deployment, RunningService, edge_key


SPACE_COUNT = 10
AIR_COUNT = 30
GROUND_COUNT = 60
SAGIN_NODE_COUNT = SPACE_COUNT + AIR_COUNT + GROUND_COUNT


@dataclass
class ScenarioConfig:
    name: str = "toy"
    physical_nodes: int = 6
    virtual_nodes: int = 3

    def validate(self) -> None:
        if self.name not in {"toy", "mobility", "congestion", "outage", "compound", "sagin100"}:
            raise ValueError("scenario must be toy, mobility, congestion, outage, compound, or sagin100")
        if self.name == "sagin100" and self.physical_nodes != SAGIN_NODE_COUNT:
            raise ValueError("sagin100 requires physical_nodes=100 (10 Space + 30 Air + 60 Ground)")
        if self.physical_nodes < max(6, self.virtual_nodes + 2):
            raise ValueError("physical_nodes must support the requested virtual network")
        if not 2 <= self.virtual_nodes <= 10:
            raise ValueError("virtual_nodes must be between 2 and 10")


class DynamicSAGINScenario:
    def __init__(self, config: ScenarioConfig, seed: int):
        config.validate()
        self.config = config
        self.seed = seed

    def _edges(self) -> List[Tuple[int, int]]:
        if self.config.name == "sagin100":
            edges = set()
            for node_id in range(SPACE_COUNT):
                edges.add(edge_key(node_id, (node_id + 1) % SPACE_COUNT))
                edges.add(edge_key(node_id, (node_id + 2) % SPACE_COUNT))
            for offset in range(AIR_COUNT):
                node_id = SPACE_COUNT + offset
                edges.add(edge_key(node_id, SPACE_COUNT + (offset + 1) % AIR_COUNT))
                edges.add(edge_key(node_id, SPACE_COUNT + (offset + 3) % AIR_COUNT))
            for offset in range(GROUND_COUNT):
                node_id = SPACE_COUNT + AIR_COUNT + offset
                if offset % 10 != 9:
                    edges.add(edge_key(node_id, node_id + 1))
                if offset < 50:
                    edges.add(edge_key(node_id, node_id + 10))
            for satellite in range(SPACE_COUNT):
                for offset in range(0, AIR_COUNT, 3):
                    edges.add(edge_key(satellite, SPACE_COUNT + offset))
            for offset in range(AIR_COUNT):
                for ground_offset in range(offset % 10, GROUND_COUNT, 20):
                    edges.add(edge_key(SPACE_COUNT + offset, SPACE_COUNT + AIR_COUNT + ground_offset))
            return sorted(edges)
        node_count = self.config.physical_nodes
        edges = {(index, (index + 1) % node_count) for index in range(node_count)}
        edges.update((index, index + 2) for index in range(node_count - 2))
        return sorted((min(u, v), max(u, v)) for u, v in edges)

    def snapshot(self, step: int) -> nx.Graph:
        if self.config.name == "sagin100":
            return self._sagin100_snapshot(step)
        rng = random.Random(self.seed * 10_000 + step)
        graph = nx.Graph()
        intensity = {"toy": 0.7, "mobility": 1.0, "congestion": 1.25, "outage": 1.0, "compound": 1.35}[self.config.name]
        for node_id in range(self.config.physical_nodes):
            mobility = rng.uniform(-4.0, 4.0) * intensity
            cpu = max(6.0, 90.0 - 4.0 * node_id - 4.0 * step * intensity + mobility)
            queue = min(1.0, 0.07 * step * intensity + 0.025 * node_id + rng.uniform(0.0, 0.05))
            energy = max(0.03, 1.0 - 0.035 * step * intensity - 0.02 * node_id)
            if self.config.name in {"congestion", "compound"} and step >= 4 and node_id in {2, 3}:
                cpu = max(4.0, cpu - 30.0)
                queue = min(1.0, queue + 0.45)
            if self.config.name in {"mobility", "compound"} and node_id < 2:
                energy = max(0.03, energy - 0.12 * (step % 3))
            node_fault = 1.0 if self.config.name == "compound" and 6 <= step <= 7 and node_id == 3 else 0.0
            if node_fault:
                cpu = min(cpu, 5.0)
                queue = 1.0
                energy = min(energy, 0.05)
            graph.add_node(node_id, cpu=cpu, max_cpu=100.0, storage=50.0, energy=energy, queue=queue, fault=node_fault, domain=0 if node_id < 2 else (1 if node_id < 4 else 2))

        for u, v in self._edges():
            bandwidth = max(2.0, 94.0 - 5.0 * step * intensity - 1.8 * (u + v) + rng.uniform(-5.0, 5.0))
            delay = 2.0 + abs(u - v) + 0.25 * step * intensity
            loss = min(1.0, 0.02 * step * intensity + 0.01 * abs(u - v))
            visible_time = max(0.05, 1.7 - 0.10 * step * intensity - 0.06 * abs(u - v))
            is_event_link = (u, v) in {(2, 3), (1, 3)}
            link_fault = 0.0
            if self.config.name in {"outage", "compound"} and 5 <= step <= 7 and is_event_link:
                bandwidth = 0.0
                delay += 12.0
                loss = 1.0
                visible_time = 0.01
                link_fault = 1.0
            graph.add_edge(u, v, bandwidth=bandwidth, max_bandwidth=100.0, delay=delay, loss=loss, visible_time=visible_time, fault=link_fault)
        return graph

    @staticmethod
    def _domain(node_id: int) -> str:
        if node_id < SPACE_COUNT:
            return "space"
        if node_id < SPACE_COUNT + AIR_COUNT:
            return "air"
        return "ground"

    def _sagin100_snapshot(self, step: int) -> nx.Graph:
        """Create a 10/30/60 SAGIN with effective links changing by visibility and faults."""
        rng = random.Random(self.seed * 10_000 + step)
        graph = nx.Graph()
        node_profiles = {"space": (180.0, 0.82, 0.12, 0), "air": (105.0, 0.68, 0.22, 1), "ground": (260.0, 1.0, 0.06, 2)}
        for node_id in range(SAGIN_NODE_COUNT):
            domain = self._domain(node_id)
            max_cpu, base_energy, base_queue, domain_index = node_profiles[domain]
            mobility = 0.15 if domain == "space" else (0.35 if domain == "air" else 0.03)
            cpu = max(8.0, max_cpu * (0.82 - 0.012 * step) + rng.uniform(-12.0, 12.0))
            queue = min(1.0, max(0.0, base_queue + 0.02 * step + rng.uniform(-0.05, 0.05)))
            energy = max(0.03, base_energy - mobility * (0.25 + 0.5 * (1.0 + math.sin(0.23 * step + node_id))))
            fault = 1.0 if 6 <= step % 18 <= 7 and node_id in {3, SPACE_COUNT + 6} else 0.0
            if fault:
                cpu, queue, energy = min(cpu, 5.0), 1.0, min(energy, 0.03)
            graph.add_node(node_id, cpu=cpu, max_cpu=max_cpu, storage=120.0 if domain == "ground" else 70.0, energy=energy, queue=queue, fault=fault, available=1.0 - fault, domain=domain_index)

        profiles = {
            ("space", "space"): (95.0, 24.0, 0.015), ("air", "air"): (65.0, 7.0, 0.025),
            ("ground", "ground"): (180.0, 2.0, 0.005), ("air", "space"): (45.0, 32.0, 0.045),
            ("air", "ground"): (75.0, 10.0, 0.030),
        }
        for u, v in self._edges():
            pair = tuple(sorted((self._domain(u), self._domain(v))))
            capacity, base_delay, base_loss = profiles[pair]
            mobile = pair != ("ground", "ground")
            visible = 1.0 if not mobile else max(0.0, 0.5 + 0.5 * math.sin(0.19 * step + 0.17 * u + 0.11 * v))
            fault = 1.0 if visible < 0.12 or (5 <= step % 18 <= 7 and edge_key(u, v) in {edge_key(2, 3), edge_key(1, 3)}) else 0.0
            bandwidth = 0.0 if fault else max(2.0, capacity * (0.62 + 0.32 * visible) - 1.2 * step + rng.uniform(-5.0, 5.0))
            graph.add_edge(u, v, bandwidth=bandwidth, max_bandwidth=capacity, delay=base_delay + (1.0 - visible) * base_delay * 0.55 + rng.uniform(0.0, 1.5), loss=1.0 if fault else min(0.95, base_loss + 0.08 * (1.0 - visible)), visible_time=0.0 if fault else max(0.05, 2.0 * visible), fault=fault)
        return graph

    def initial_history(self, window: int = 4) -> List[nx.Graph]:
        return [self.snapshot(step) for step in range(window)]

    def build_service(self, service_id: str = "svc-001", virtual_nodes: int = None, arrival_time: int = 0) -> RunningService:
        if self.config.name == "sagin100":
            rng = random.Random(self.seed * 1_000_003 + arrival_time * 97 + sum(map(ord, service_id)))
            count = virtual_nodes if virtual_nodes is not None else self.config.virtual_nodes
            if not 2 <= count <= 10:
                raise ValueError("virtual_nodes must be between 2 and 10")
            virtual_graph = nx.Graph()
            for node_id in range(count):
                virtual_graph.add_node(node_id, cpu=rng.uniform(6.0, 18.0), type="vnf")
            for node_id in range(count - 1):
                virtual_graph.add_edge(node_id, node_id + 1, bandwidth=rng.uniform(3.0, 12.0))
            if count >= 5 and rng.random() < 0.45:
                virtual_graph.add_edge(0, count - 1, bandwidth=rng.uniform(3.0, 8.0))
            return RunningService(
                service_id=service_id,
                virtual_graph=virtual_graph,
                deployment=Deployment(description="Awaiting admission"),
                remaining_lifetime=rng.randint(8, 24),
                max_delay=rng.uniform(35.0, 110.0),
                priority=rng.choice((0.7, 1.0, 1.4)),
            )
        virtual_graph = nx.Graph()
        for node_id in range(self.config.virtual_nodes):
            virtual_graph.add_node(node_id, cpu=16.0 + 3.0 * (node_id % 3))
        for node_id in range(self.config.virtual_nodes - 1):
            virtual_graph.add_edge(node_id, node_id + 1, bandwidth=14.0 + 2.0 * (node_id % 2))
        node_mapping = {node_id: node_id + 1 for node_id in virtual_graph.nodes}
        link_mapping = {
            (node_id, node_id + 1): [edge_key(node_id + 1, node_id + 2)]
            for node_id in range(self.config.virtual_nodes - 1)
        }
        return RunningService("svc-001", virtual_graph, Deployment(node_mapping, link_mapping, "Current deployment"), 10, 18.0, 1.0)


def build_physical_snapshot(step: int) -> nx.Graph:
    """Backward-compatible toy scenario function for older scripts."""
    return DynamicSAGINScenario(ScenarioConfig(), seed=7).snapshot(step)


def build_running_service() -> RunningService:
    """Backward-compatible default service builder."""
    return DynamicSAGINScenario(ScenarioConfig(), seed=7).build_service()
