from dataclasses import dataclass
from typing import Optional


@dataclass
class PredictorConfig:
    history_window: int = 4
    future_horizon: int = 3
    hidden_dim: int = 64
    # A randomly initialized predictor must never be used as an experiment model.
    use_learned_model: bool = False
    device: str = "cpu"
    checkpoint_path: Optional[str] = None
    require_checkpoint: bool = False


@dataclass
class PlanningConfig:
    node_risk_threshold: float = 0.65
    link_risk_threshold: float = 0.70
    full_migration_threshold: float = 0.82
    partial_ratio_threshold: float = 0.35
    max_trigger_delay: int = 3
    risk_weight: float = 1.0
    migration_cost_weight: float = 0.7
    continuity_weight: float = 0.9


@dataclass
class ExecutionConfig:
    interruption_cost_weight: float = 1.0
    risk_weight: float = 1.0
    fragmentation_weight: float = 0.4
    migration_cost_weight: float = 0.8
    node_migration_cost_weight: float = 0.6
    link_reroute_cost_weight: float = 0.4
    min_risk_improvement: float = 0.01
    enforce_risk_reduction: bool = True
