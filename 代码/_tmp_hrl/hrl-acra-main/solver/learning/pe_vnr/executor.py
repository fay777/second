import copy

from base import Solution


class ReconfigurationExecutor:
    def __init__(self, controller, shortest_method="bfs_shortest", k_shortest=10):
        self.controller = controller
        self.shortest_method = shortest_method
        self.k_shortest = k_shortest

    def _build_candidate_order(self, p_net, node_risk):
        all_nodes = list(p_net.nodes)
        return sorted(all_nodes, key=lambda node_id: (node_risk.get(node_id, 0.0), -sum(p_net.nodes[node_id].get(attr.name, 0.0) for attr in self.controller.node_resource_attrs)))

    def _score_solution(self, solution, node_risk, link_risk):
        if not solution["result"]:
            return float("-inf")
        node_score = 0.0
        for _, p_node_id in solution["node_slots"].items():
            node_score += 1.0 - node_risk.get(p_node_id, 0.0)
        link_score = 0.0
        routed_links = 0
        for _, p_links in solution["link_paths"].items():
            for p_link in p_links:
                link_score += 1.0 - link_risk.get(p_link, 0.0)
                routed_links += 1
        return node_score + (link_score / max(1, routed_links))

    def _prepare_released_network(self, v_net, p_net, current_solution):
        working_p_net = copy.deepcopy(p_net)
        if current_solution is not None and len(current_solution["node_slots"]) > 0:
            self.controller.release(v_net, working_p_net, current_solution)
        return working_p_net

    def _attempt_mapping(self, v_net, p_net, node_order, fixed_node_slots):
        temp_p_net = copy.deepcopy(p_net)
        solution = Solution(v_net)
        used_p_nodes = []

        for v_node_id, p_node_id in fixed_node_slots.items():
            place_result, _ = self.controller.place(v_net, temp_p_net, v_node_id, p_node_id, solution)
            if not place_result:
                return solution
            used_p_nodes.append(p_node_id)

        for v_node_id in v_net.nodes:
            if v_node_id in fixed_node_slots:
                continue
            candidate_nodes = self.controller.find_candidate_nodes(v_net, temp_p_net, v_node_id, filter=used_p_nodes)
            candidate_nodes = [node_id for node_id in node_order if node_id in candidate_nodes]
            if len(candidate_nodes) == 0:
                solution["place_result"] = False
                solution["result"] = False
                return solution
            place_result, _ = self.controller.place(v_net, temp_p_net, v_node_id, candidate_nodes[0], solution)
            if not place_result:
                solution["place_result"] = False
                solution["result"] = False
                return solution
            used_p_nodes.append(candidate_nodes[0])

        route_result = self.controller.link_mapping(
            v_net,
            temp_p_net,
            solution,
            sorted_v_links=None,
            shortest_method=self.shortest_method,
            k=self.k_shortest,
            inplace=True,
        )
        solution["route_result"] = route_result
        solution["result"] = route_result
        return solution

    def execute(self, v_net, p_net, current_solution, plan, risk_prediction):
        if current_solution is None:
            raise ValueError("current_solution is required for reconfiguration.")

        if not plan.migrate:
            stable_solution = copy.deepcopy(current_solution)
            stable_solution["description"] = "Keep Current Deployment"
            return stable_solution

        released_p_net = self._prepare_released_network(v_net, p_net, current_solution)
        candidate_order = self._build_candidate_order(released_p_net, risk_prediction.node_risk)

        if plan.scope == "link-only":
            fixed_node_slots = dict(current_solution["node_slots"])
        elif plan.scope == "partial":
            fixed_node_slots = {
                v_node_id: p_node_id
                for v_node_id, p_node_id in current_solution["node_slots"].items()
                if v_node_id not in set(plan.target_v_nodes)
            }
        else:
            fixed_node_slots = {}

        best_solution = self._attempt_mapping(v_net, released_p_net, candidate_order, fixed_node_slots)
        best_score = self._score_solution(best_solution, risk_prediction.node_risk, risk_prediction.link_risk)

        reverse_order = list(reversed(candidate_order))
        alternative_solution = self._attempt_mapping(v_net, released_p_net, reverse_order, fixed_node_slots)
        alternative_score = self._score_solution(alternative_solution, risk_prediction.node_risk, risk_prediction.link_risk)
        if alternative_score > best_score:
            best_solution = alternative_solution

        best_solution["description"] = f"Proactive Reconfiguration ({plan.scope})"
        best_solution["migration_plan"] = {
            "scope": plan.scope,
            "trigger_delay": plan.trigger_delay,
            "estimated_cost": plan.estimated_cost,
            "risk_score": plan.risk_score,
        }
        return best_solution
