"""Dependency-free smoke checks for the stateless multi-service HAC worker."""

import networkx as nx

from pe_vnr.config import ExecutionConfig
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.hac_worker import MultiServiceHACWorkerAdapter, PendingHACDecision
from pe_vnr.topology import Deployment, RiskPrediction, RunningService, edge_key


def build_graph(nodes=4):
    graph = nx.complete_graph(nodes)
    for node_id in graph.nodes:
        graph.nodes[node_id].update(cpu=100.0, max_cpu=100.0, queue=0.0, energy=1.0, fault=0.0, available=1.0, domain=0)
    for u, v in graph.edges:
        graph.edges[(u, v)].update(bandwidth=100.0, max_bandwidth=100.0, delay=1.0, loss=0.0, visible_time=1.0, fault=0.0)
    return graph


def build_service():
    virtual_graph = nx.Graph()
    virtual_graph.add_node(0, cpu=10.0)
    virtual_graph.add_node(1, cpu=10.0)
    virtual_graph.add_edge(0, 1, bandwidth=5.0)
    return RunningService(
        service_id="smoke-service",
        virtual_graph=virtual_graph,
        deployment=Deployment({0: 0, 1: 1}, {(0, 1): [edge_key(0, 1)]}, "initial"),
        remaining_lifetime=8,
        max_delay=10.0,
    )


def prediction(node_risk, link_risk):
    return RiskPrediction(node_risk, link_risk, [dict(node_risk)] * 3, [dict(link_risk)] * 3)


def graph_state(graph):
    return (
        {node: dict(attrs) for node, attrs in graph.nodes(data=True)},
        {edge_key(u, v): dict(attrs) for u, v, attrs in graph.edges(data=True)},
    )


def worker(upper_action):
    executor = ElasticReconfigurationExecutor(ExecutionConfig(min_risk_improvement=0.01))
    return MultiServiceHACWorkerAdapter(executor, upper_policy=lambda _: upper_action)


def test_keep():
    service, graph = build_service(), build_graph()
    adapter = worker(0)
    risks = prediction({0: 0.9, 1: 0.8, 2: 0.1, 3: 0.2}, {edge_key(0, 1): 0.9})
    before = graph_state(graph)
    decision = adapter.upper_decide(service, risks, graph)
    assert not decision.migrate and decision.scope == "none"
    assert graph_state(graph) == before and service.migration_count == 0


def test_shared_resource_pressure():
    service, physical_graph = build_service(), build_graph()
    residual_graph = build_graph()
    for node_id in residual_graph.nodes:
        residual_graph.nodes[node_id]["cpu"] = 20.0
    for edge in residual_graph.edges:
        residual_graph.edges[edge]["bandwidth"] = 20.0
    adapter = worker(0)
    risks = prediction({0: 0.9, 1: 0.8, 2: 0.1, 3: 0.2}, {edge_key(0, 1): 0.9})
    idle = adapter.upper_observation(service, physical_graph, risks, physical_graph)
    pressured = adapter.upper_observation(service, residual_graph, risks, physical_graph)
    assert pressured.state.size == idle.state.size
    assert pressured.state[-2] > idle.state[-2] and pressured.state[-1] > idle.state[-1]


def test_immediate_full():
    service, graph = build_service(), build_graph()
    adapter = worker(adapter_action(adapter_scope="full", delay=0))
    risks = prediction(
        {0: 0.9, 1: 0.8, 2: 0.1, 3: 0.2},
        {edge_key(0, 1): 0.9, edge_key(2, 3): 0.1},
    )
    decision = adapter.upper_decide(service, risks, graph)
    result = adapter.execute_now(service, decision.scope, risks, graph)
    assert decision.migrate and decision.delay == 0 and decision.scope == "full"
    assert result.outcome.migrated and result.lower_decisions == 2 and result.routing_success
    assert all(node_id in graph for node_id in result.outcome.deployment.node_mapping.values())


def test_delayed_partial_relocalizes():
    service, graph = build_service(), build_graph()
    adapter = worker(adapter_action(adapter_scope="partial", delay=2))
    at_t = prediction({0: 0.9, 1: 0.3, 2: 0.05, 3: 0.4}, {edge_key(0, 1): 0.9})
    decision = adapter.upper_decide(service, at_t, graph)
    pending = PendingHACDecision(service.service_id, 2, decision.scope, 0, 1)
    assert decision.delay == 2 and pending.scope == "partial"
    assert service.deployment.node_mapping[0] == 0
    # At due time a different host is preferable. No old target was stored.
    at_due = prediction({0: 0.9, 1: 0.3, 2: 0.4, 3: 0.05}, {edge_key(0, 1): 0.9, edge_key(1, 3): 0.1})
    result = adapter.execute_now(service, pending.scope, at_due, graph)
    assert result.outcome.migrated
    assert result.outcome.deployment.node_mapping[0] == 3


def test_failure_rolls_back():
    service, graph = build_service(), build_graph(nodes=2)
    graph.nodes[1]["fault"] = 1.0
    adapter = worker(adapter_action(adapter_scope="full", delay=0))
    risks = prediction({0: 0.9, 1: 0.8}, {edge_key(0, 1): 0.9})
    before = graph_state(graph)
    result = adapter.execute_now(service, "full", risks, graph)
    assert not result.outcome.migrated and not result.lower_success
    assert result.outcome.deployment.node_mapping == service.deployment.node_mapping
    assert graph_state(graph) == before


def adapter_action(adapter_scope, delay):
    scope_index = {"link-only": 0, "partial": 1, "full": 2}[adapter_scope]
    # PlanningEnv default action layout: 2 * (max_delay + 1) * 3.
    return (4 + delay) * 3 + scope_index


def main():
    test_keep()
    test_shared_resource_pressure()
    test_immediate_full()
    test_delayed_partial_relocalizes()
    test_failure_rolls_back()
    print("HAC worker smoke checks passed")


if __name__ == "__main__":
    main()
