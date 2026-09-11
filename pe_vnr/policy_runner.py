"""Inference runner for modular heuristic and Actor-Critic components."""

from pathlib import Path
from typing import Dict, Optional

import torch

from .dynamic_env import DynamicTopologyEnv
from .mdp.actor_critic import ExecutionActorCritic, PlanningActorCritic


def _load(model: torch.nn.Module, path: Optional[Path], device: torch.device, label: str) -> None:
    if path is None:
        raise ValueError(f"{label} AC component requires a checkpoint path.")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()


class ModularPolicyRunner:
    def __init__(
        self,
        planner_component: str,
        execution_component: str,
        planning_checkpoint: Optional[Path],
        execution_checkpoint: Optional[Path],
        device: str = "cpu",
    ):
        self.planner_component = planner_component
        self.execution_component = execution_component
        self.planning_checkpoint = planning_checkpoint
        self.execution_checkpoint = execution_checkpoint
        self.device = torch.device(device)

    def _planning_action(self, env: DynamicTopologyEnv, observation, model) -> int:
        if self.planner_component == "heuristic":
            prediction = env._training_prediction
            plan = env.planner.plan(env.service, prediction)
            if not plan.migrate:
                return env.planning_env.encode_action(0, 0, 0)
            scope_index = env.planning_env.scopes.index(plan.scope)
            delay = min(plan.trigger_delay, env.planning_env.max_delay)
            return env.planning_env.encode_action(1, delay, scope_index)
        with torch.no_grad():
            state = torch.as_tensor(observation.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(observation.action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            return int(model.act(state).masked_fill(~mask, -1e9).argmax(dim=-1).item())

    def _execution_action(self, observation, model) -> int:
        if self.execution_component == "heuristic":
            return 0
        with torch.no_grad():
            state = torch.as_tensor(observation.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            candidates = torch.as_tensor(observation.candidate_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(model.act(state, candidates).argmax(dim=-1).item())

    def run(self, env: DynamicTopologyEnv, max_planning_steps: int) -> Dict[str, object]:
        observation = env.reset()
        planning_model = None
        execution_model = None
        if self.planner_component == "ac":
            planning_model = PlanningActorCritic(observation.state.size, observation.action_mask.size).to(self.device)
            _load(planning_model, self.planning_checkpoint, self.device, "planning")
        planning_steps = 0
        while planning_steps < max_planning_steps and env.service.status == "running":
            if env.training_stage == "planning":
                action = self._planning_action(env, observation, planning_model)
                transition = env.planning_step(action)
                planning_steps += 1
            else:
                if self.execution_component == "ac" and execution_model is None:
                    execution_model = ExecutionActorCritic(
                        observation.state.size,
                        observation.candidate_features.shape[1],
                    ).to(self.device)
                    _load(execution_model, self.execution_checkpoint, self.device, "execution")
                action = self._execution_action(observation, execution_model)
                transition = env.execution_step(action)
            if transition.terminated:
                break
            observation = transition.observation
        return {"service": env.service.clone(), "metrics": env.metrics.summary(), "planning_steps": planning_steps}
