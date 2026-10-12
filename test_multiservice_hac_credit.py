import unittest
from types import SimpleNamespace

import networkx as nx
import torch

from pe_vnr.mdp.execution_env import ExecutionEnv
from pe_vnr.mdp.actor_critic import PlanningActorCritic
from pe_vnr.config import ExecutionConfig
from pe_vnr.executor import ElasticReconfigurationExecutor
from pe_vnr.hac_worker import MultiServiceHACWorkerAdapter
from pe_vnr.planner import MigrationPlan
from pe_vnr.topology import Deployment, RiskPrediction, RunningService
from train_multiservice_hac import (
    RewardWeights,
    assign_event_rewards,
    execution_diagnostics,
    event_reward,
    update_policy,
    upper_service_trajectories,
)


class UpperServiceCreditTests(unittest.TestCase):
    def test_interleaved_services_have_independent_discounted_returns(self):
        events = [
            SimpleNamespace(event_type="upper_decision", origin_decision_id=index,
                            service_id=service, time_step=index // 2 + 1)
            for index, service in enumerate(("A", "B", "A", "B"))
        ]
        model = PlanningActorCritic(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        records = []
        for reward in (1.0, 10.0, 2.0, 20.0):
            state = torch.ones(1, 4)
            records.append({
                "log_prob": torch.log_softmax(model.act(state), dim=-1)[0, 0],
                "value": model.value(state).squeeze(),
                "reward": reward,
                "action_group": "keep",
            })

        trajectories = upper_service_trajectories(events, records)
        self.assertEqual([[record["reward"] for record in group] for group in trajectories], [[1.0, 2.0], [10.0, 20.0]])

        update = update_policy(model, optimizer, trajectories, gamma=0.5)
        # A: (2, 2), B: (20, 20). No return includes another service's reward.
        self.assertAlmostEqual(update["return_mean"], 11.0)
        self.assertEqual(update["trajectories"], 2.0)

    def test_duplicate_decision_id_is_rejected(self):
        events = [
            SimpleNamespace(event_type="upper_decision", origin_decision_id=1,
                            service_id=service, time_step=1)
            for service in ("A", "B")
        ]
        with self.assertRaisesRegex(RuntimeError, "Duplicate HAC decision ID"):
            upper_service_trajectories(events, [{"reward": 1.0}, {"reward": 2.0}])

    def test_delayed_execution_reward_returns_to_origin_decision(self):
        decision = SimpleNamespace(scope="partial")
        events = [
            SimpleNamespace(
                event_type="upper_decision", origin_decision_id=index,
                service_id=service, time_step=1, execution=None, sla_violated=False,
            )
            for index, service in enumerate(("A", "B"))
        ]
        execution = SimpleNamespace(
            outcome=SimpleNamespace(
                node_risk_reduction=0.1, link_risk_reduction=0.2,
                total_cost=0.0, migrated=True,
            ),
            lower_trace=[],
        )
        events.append(SimpleNamespace(
            event_type="pending_execution", origin_decision_id=0,
            service_id="A", time_step=3, decision=decision,
            execution=execution, sla_violated=False, post_sla_violated=False,
        ))
        records = [{"reward": 0.0}, {"reward": 0.0}]

        assign_event_rewards(events, {}, records, [], RewardWeights(), gamma=0.5)

        self.assertAlmostEqual(records[0]["reward"], -0.1)
        self.assertAlmostEqual(sum(records[0]["reward_components"].values()), records[0]["reward"])
        self.assertEqual(records[1]["reward"], 0.0)

    def test_future_sla_credit_stops_at_next_decision(self):
        events = [
            SimpleNamespace(
                event_type="upper_decision", origin_decision_id=index,
                service_id="A", time_step=time_step, execution=None, sla_violated=False,
            )
            for index, time_step in enumerate((1, 2))
        ]
        records = [{"reward": 0.0}, {"reward": 0.0}]
        history = {(2, "A"): True, (3, "A"): False, (4, "A"): False}

        assign_event_rewards(events, history, records, [], RewardWeights())

        self.assertEqual([record["reward"] for record in records], [-1.0, 0.0])

    def test_keep_pays_same_current_sla_penalty_as_unrepaired_migration(self):
        keep = SimpleNamespace(event_type="upper_decision", execution=None, sla_violated=True)
        migration = SimpleNamespace(
            execution=SimpleNamespace(outcome=SimpleNamespace(
                node_risk_reduction=0.0, link_risk_reduction=0.0,
                total_cost=0.0, migrated=True,
            )),
            sla_violated=True, post_sla_violated=True,
            decision=SimpleNamespace(scope="partial"),
        )

        _, keep_components, _ = event_reward(keep, RewardWeights())
        _, migration_components, _ = event_reward(migration, RewardWeights())

        self.assertEqual(keep_components["sla"], -1.0)
        self.assertEqual(migration_components["sla"], -1.0)

    def test_stale_pending_cancellation_penalizes_originating_upper_action(self):
        cancelled = SimpleNamespace(
            event_type="pending_cancelled",
            execution=None,
            sla_violated=False,
        )

        reward, components, _ = event_reward(cancelled, RewardWeights())

        self.assertEqual(components["failure"], -1.0)
        self.assertEqual(reward, -1.0)

    def test_discount_uses_elapsed_slots_for_same_service(self):
        events = [
            SimpleNamespace(event_type="upper_decision", origin_decision_id=index,
                            service_id=service, time_step=time_step)
            for index, (service, time_step) in enumerate((("A", 1), ("B", 2), ("A", 4)))
        ]
        model = PlanningActorCritic(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        records = []
        for reward in (1.0, 10.0, 2.0):
            state = torch.ones(1, 4)
            records.append({
                "log_prob": torch.log_softmax(model.act(state), dim=-1)[0, 0],
                "value": model.value(state).squeeze(),
                "reward": reward,
            })

        update = update_policy(model, optimizer, upper_service_trajectories(events, records), gamma=0.5)

        self.assertAlmostEqual(update["return_mean"], (1.25 + 2.0 + 10.0) / 3.0, places=6)

    def test_repeated_critic_updates_reduce_training_error(self):
        torch.manual_seed(4)
        model = PlanningActorCritic(4, 2)
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.0)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=0.01)
        trajectories = []
        for reward in (-5.0, -5.0, -1.0, -1.0):
            state = torch.ones(1, 4)
            trajectories.append([{
                "state": state.squeeze(0),
                "log_prob": torch.log_softmax(model.act(state), dim=-1)[0, 0],
                "value": model.value(state).squeeze().detach(),
                "reward": reward,
            }])

        update = update_policy(
            model, actor_optimizer, trajectories, gamma=0.99,
            critic_optimizer=critic_optimizer, critic_updates=10,
        )

        self.assertLess(update["critic_post_rmse"], update["critic_rmse"])

    def test_candidate_failure_diagnostics_distinguish_initial_and_late(self):
        def failed(counts):
            execution = SimpleNamespace(
                failure_stage="no_candidate_host",
                candidate_counts=counts,
                outcome=SimpleNamespace(migrated=False),
            )
            return SimpleNamespace(execution=execution, decision=SimpleNamespace(scope="partial"))

        diagnostics = execution_diagnostics([failed([]), failed([3, 1])])

        self.assertEqual(diagnostics["no_candidate_initial"], 1)
        self.assertEqual(diagnostics["no_candidate_after_placement"], 1)

    def test_empty_lower_batch_has_complete_gate_metrics(self):
        model = PlanningActorCritic(4, 2)
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.001)

        update = update_policy(model, optimizer, [], gamma=0.99)

        self.assertEqual(update["records"], 0.0)
        self.assertEqual(update["gate_keep_sla_mean"], 0.0)
        self.assertEqual(update["gate_migrate_failure_mean"], 0.0)

    def test_lookahead_masks_host_that_strands_remaining_vnf(self):
        physical = nx.complete_graph(4)
        for node_id, cpu in {0: 8.0, 1: 4.0, 2: 0.0, 3: 0.0}.items():
            physical.nodes[node_id].update(cpu=cpu, max_cpu=8.0, energy=1.0, available=1.0, fault=0.0)
        for _, _, attrs in physical.edges(data=True):
            attrs.update(bandwidth=100.0, visible_time=1.0, fault=0.0)
        virtual = nx.Graph()
        virtual.add_node(0, cpu=4.0)
        virtual.add_node(1, cpu=8.0)
        virtual.add_edge(0, 1, bandwidth=1.0)
        service = RunningService(
            "svc", virtual, Deployment({0: 3, 1: 2}, {(0, 1): [(2, 3)]}), 5, 10.0
        )
        plan = MigrationPlan(True, 0, "full", [0, 1], [(0, 1)], 1.0, 0.0)

        observation = ExecutionEnv().reset(physical, service, plan, {})

        self.assertEqual(observation.candidate_nodes, [1])
        self.assertTrue(ExecutionEnv.can_start(physical, service, plan, {}))

    def test_lower_step_reserves_completed_virtual_link_bandwidth(self):
        physical = nx.path_graph(4)
        for node_id in physical.nodes:
            physical.nodes[node_id].update(cpu=8.0, max_cpu=8.0, energy=1.0, available=1.0, fault=0.0)
        for _, _, attrs in physical.edges(data=True):
            attrs.update(bandwidth=5.0, visible_time=1.0, fault=0.0, delay=1.0, loss=0.0)
        virtual = nx.Graph()
        virtual.add_node(0, cpu=2.0)
        virtual.add_node(1, cpu=2.0)
        virtual.add_edge(0, 1, bandwidth=5.0)
        service = RunningService(
            "svc", virtual, Deployment({0: 2, 1: 3}, {(0, 1): [(2, 3)]}), 5, 10.0
        )
        plan = MigrationPlan(True, 0, "full", [0, 1], [(0, 1)], 1.0, 0.0)
        environment = ExecutionEnv()

        first = environment.reset(physical, service, plan, {})
        self.assertEqual(first.candidate_nodes[0], 0)
        second, done, _ = environment.step(0)
        self.assertFalse(done)
        self.assertEqual(second.candidate_nodes[0], 1)
        _, done, _ = environment.step(0)

        self.assertTrue(done)
        self.assertEqual(environment.link_mapping[(0, 1)], [(0, 1)])
        self.assertEqual(environment.graph.edges[(0, 1)]["bandwidth"], 0.0)

    def test_upper_masks_only_immediate_node_remap_without_starting_host(self):
        physical = nx.Graph()
        physical.add_node(0, cpu=4.0, max_cpu=4.0, energy=1.0, available=1.0, fault=0.0)
        virtual = nx.Graph()
        virtual.add_node(0, cpu=4.0)
        service = RunningService("svc", virtual, Deployment({0: 0}, {}), 5, 10.0)
        prediction = RiskPrediction({0: 0.5}, {}, [{0: 0.5}], [{}])
        worker = MultiServiceHACWorkerAdapter(ElasticReconfigurationExecutor(ExecutionConfig()))

        observation = worker.upper_observation(service, physical, prediction)
        partial = worker.planning_env.scopes.index("partial")
        full = worker.planning_env.scopes.index("full")

        self.assertFalse(observation.action_mask[worker.planning_env.encode_action(1, 0, partial)])
        self.assertFalse(observation.action_mask[worker.planning_env.encode_action(1, 0, full)])
        self.assertTrue(observation.action_mask[worker.planning_env.encode_action(1, 1, partial)])
        self.assertFalse(worker.pending_scope_still_feasible(service, "partial", prediction, physical))
        self.assertFalse(worker.pending_scope_still_feasible(service, "full", prediction, physical))

    def test_upper_immediate_full_mask_uses_released_service_resources(self):
        physical = nx.Graph()
        for node_id in (0, 1):
            # Both hosts are fully occupied by the service before its transaction.
            physical.add_node(node_id, cpu=0.0, max_cpu=4.0, energy=1.0, available=1.0, fault=0.0)
        virtual = nx.Graph()
        virtual.add_node(0, cpu=4.0)
        virtual.add_node(1, cpu=4.0)
        service = RunningService("svc", virtual, Deployment({0: 0, 1: 1}, {}), 5, 10.0)
        prediction = RiskPrediction({0: 0.5, 1: 0.5}, {}, [{0: 0.5, 1: 0.5}], [{}])
        worker = MultiServiceHACWorkerAdapter(ElasticReconfigurationExecutor(ExecutionConfig()))

        observation = worker.upper_observation(service, physical, prediction)
        full = worker.planning_env.scopes.index("full")

        self.assertTrue(observation.action_mask[worker.planning_env.encode_action(1, 0, full)])

    def test_upper_masks_full_when_partial_reconfiguration_can_start(self):
        physical = nx.complete_graph(4)
        for node_id in physical.nodes:
            physical.nodes[node_id].update(cpu=4.0, max_cpu=4.0, energy=1.0, available=1.0, fault=0.0)
        virtual = nx.Graph()
        virtual.add_node(0, cpu=2.0)
        virtual.add_node(1, cpu=2.0)
        virtual.add_edge(0, 1, bandwidth=1.0)
        service = RunningService(
            "svc", virtual, Deployment({0: 0, 1: 1}, {(0, 1): [(0, 1)]}), 5, 10.0
        )
        for _, _, attrs in physical.edges(data=True):
            attrs.update(bandwidth=10.0, visible_time=1.0, fault=0.0)
        prediction = RiskPrediction(
            {node_id: 0.5 for node_id in physical.nodes},
            {(u, v): 0.5 for u, v in physical.edges},
            [{node_id: 0.5 for node_id in physical.nodes}],
            [{(u, v): 0.5 for u, v in physical.edges}],
        )
        worker = MultiServiceHACWorkerAdapter(ElasticReconfigurationExecutor(ExecutionConfig()))

        observation = worker.upper_observation(service, physical, prediction)
        partial = worker.planning_env.scopes.index("partial")
        full = worker.planning_env.scopes.index("full")

        self.assertTrue(observation.action_mask[worker.planning_env.encode_action(1, 0, partial)])
        self.assertFalse(observation.action_mask[worker.planning_env.encode_action(1, 0, full)])

    def test_link_only_is_masked_when_dry_run_cannot_route(self):
        physical = nx.Graph()
        for node_id in (0, 1):
            physical.add_node(node_id, cpu=4.0, max_cpu=4.0, energy=1.0, available=1.0, fault=0.0)
        physical.add_edge(0, 1, bandwidth=10.0, max_bandwidth=10.0, visible_time=0.0, fault=0.0)
        virtual = nx.Graph()
        virtual.add_node(0, cpu=1.0)
        virtual.add_node(1, cpu=1.0)
        virtual.add_edge(0, 1, bandwidth=1.0)
        service = RunningService(
            "svc", virtual, Deployment({0: 0, 1: 1}, {(0, 1): [(0, 1)]}), 5, 10.0
        )
        prediction = RiskPrediction({0: 0.5, 1: 0.5}, {(0, 1): 0.5}, [{0: 0.5, 1: 0.5}], [{(0, 1): 0.5}])
        worker = MultiServiceHACWorkerAdapter(ElasticReconfigurationExecutor(ExecutionConfig()))

        observation = worker.upper_observation(service, physical, prediction)
        link_only = worker.planning_env.scopes.index("link-only")

        self.assertFalse(observation.action_mask[worker.planning_env.encode_action(1, 0, link_only)])
        self.assertFalse(worker.pending_scope_still_feasible(service, "link-only", prediction, physical))


if __name__ == "__main__":
    unittest.main()
