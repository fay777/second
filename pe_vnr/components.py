"""Component presets and environment assembly for modular experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import ExecutionConfig, PlanningConfig, PredictorConfig
from .dynamic_env import DynamicTopologyEnv
from .executor import ElasticReconfigurationExecutor
from .planner import MigrationPlanner
from .risk_predictor import STRiskPredictor
from .scenarios import DynamicSAGINScenario, ScenarioConfig
from .topology import TimeVaryingTopology


MODE_PRESETS: Dict[str, Dict[str, str]] = {
    "heuristic-proactive": {"predictor": "heuristic", "planner": "heuristic", "execution": "heuristic"},
    "ac-proactive": {"predictor": "heuristic", "planner": "ac", "execution": "ac"},
    "stgcn-heuristic": {"predictor": "stgcn", "planner": "heuristic", "execution": "heuristic"},
    "full-model": {"predictor": "stgcn", "planner": "ac", "execution": "ac"},
}


@dataclass
class ComponentConfig:
    mode: str = "heuristic-proactive"
    predictor: Optional[str] = None
    planner: Optional[str] = None
    execution: Optional[str] = None
    predictor_checkpoint: Optional[Path] = None
    planning_checkpoint: Optional[Path] = None
    execution_checkpoint: Optional[Path] = None
    seed: int = 7
    device: str = "cpu"
    scenario: str = "toy"
    physical_nodes: int = 6
    virtual_nodes: int = 3
    history_window: int = 4
    future_horizon: int = 3

    def resolved(self) -> Dict[str, str]:
        if self.mode not in MODE_PRESETS:
            raise ValueError(f"Unknown mode: {self.mode}")
        result = dict(MODE_PRESETS[self.mode])
        for name in ("predictor", "planner", "execution"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if result["predictor"] not in {"heuristic", "stgcn"}:
            raise ValueError("predictor must be heuristic or stgcn")
        if result["planner"] not in {"heuristic", "ac"} or result["execution"] not in {"heuristic", "ac"}:
            raise ValueError("planner and execution must be heuristic or ac")
        return result


def build_environment(config: ComponentConfig) -> DynamicTopologyEnv:
    resolved = config.resolved()
    scenario = DynamicSAGINScenario(
        ScenarioConfig(config.scenario, config.physical_nodes, config.virtual_nodes),
        config.seed,
    )
    topology = TimeVaryingTopology(history=scenario.initial_history())
    predictor = STRiskPredictor(
        PredictorConfig(
            use_learned_model=resolved["predictor"] == "stgcn",
            checkpoint_path=str(config.predictor_checkpoint) if config.predictor_checkpoint else None,
            require_checkpoint=resolved["predictor"] == "stgcn",
            history_window=config.history_window,
            future_horizon=config.future_horizon,
            device=config.device,
        )
    )
    planner = MigrationPlanner(
        PlanningConfig(
            node_risk_threshold=0.50,
            link_risk_threshold=0.52,
            full_migration_threshold=0.72,
            partial_ratio_threshold=0.34,
            max_trigger_delay=2,
        )
    )
    return DynamicTopologyEnv(
        topology=topology,
        service=scenario.build_service(),
        topology_builder=scenario.snapshot,
        predictor=predictor,
        planner=planner,
        executor=ElasticReconfigurationExecutor(ExecutionConfig()),
    )
