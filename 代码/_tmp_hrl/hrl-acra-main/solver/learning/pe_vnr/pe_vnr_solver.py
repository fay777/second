import copy

from base import Solution
from solver.heuristic.node_rank import NRMRankSolver
from solver.solver import Solver

from .executor import ReconfigurationExecutor
from .planner import MigrationPlanner
from .risk_predictor import STRiskPredictor


class ProactiveElasticVNRSolver(Solver):
    name = "pe_vnr"

    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.fallback_solver = NRMRankSolver(controller, recorder, counter, **kwargs)
        self.risk_predictor = STRiskPredictor(
            controller,
            hidden_dim=kwargs.get("risk_hidden_dim", 64),
            use_learned_model=kwargs.get("use_learned_risk_predictor", False),
            device="cpu",
        )
        self.planner = MigrationPlanner(
            node_risk_threshold=kwargs.get("node_risk_threshold", 0.65),
            link_risk_threshold=kwargs.get("link_risk_threshold", 0.70),
            full_migration_threshold=kwargs.get("full_migration_threshold", 0.80),
            partial_ratio_threshold=kwargs.get("partial_ratio_threshold", 0.35),
            max_trigger_delay=kwargs.get("max_trigger_delay", 3),
        )
        self.executor = ReconfigurationExecutor(
            controller,
            shortest_method=self.shortest_method,
            k_shortest=self.k_shortest,
        )

    def _build_default_history(self, p_net, current_solution):
        if current_solution is None or len(current_solution["node_slots"]) == 0:
            return [copy.deepcopy(p_net)]
        released_p_net = copy.deepcopy(p_net)
        self.controller.release(current_solution["v_net"], released_p_net, current_solution)
        return [released_p_net, copy.deepcopy(p_net)]

    def solve(self, instance):
        v_net = instance["v_net"]
        p_net = instance["p_net"]
        current_solution = instance.get("current_solution")
        if current_solution is None:
            current_solution = instance.get("running_solution")

        if current_solution is None or len(current_solution["node_slots"]) == 0:
            initial_solution = self.fallback_solver.solve({"v_net": v_net, "p_net": p_net})
            initial_solution["description"] = "Initial Stable Deployment"
            return initial_solution

        current_solution = copy.deepcopy(current_solution)
        current_solution["v_net"] = v_net

        history_p_nets = instance.get("history_p_nets")
        if history_p_nets is None or len(history_p_nets) == 0:
            history_p_nets = self._build_default_history(p_net, current_solution)

        risk_prediction = self.risk_predictor.predict(history_p_nets)
        plan = self.planner.plan(v_net, current_solution, risk_prediction)

        if not plan.migrate:
            stable_solution = copy.deepcopy(current_solution)
            stable_solution["result"] = True
            stable_solution["description"] = "Stable Deployment Without Reconfiguration"
            stable_solution["migration_plan"] = {
                "scope": "none",
                "trigger_delay": 0,
                "estimated_cost": 0.0,
                "risk_score": plan.risk_score,
            }
            return stable_solution

        executed_solution = self.executor.execute(v_net, p_net, current_solution, plan, risk_prediction)
        executed_solution["risk_prediction"] = {
            "mean_node_risk": risk_prediction.mean_node_risk,
            "mean_link_risk": risk_prediction.mean_link_risk,
            "max_node_risk": risk_prediction.max_node_risk,
            "max_link_risk": risk_prediction.max_link_risk,
        }
        return executed_solution
