"""Trajectory-level ST-GCN datasets and horizon-wise evaluation metrics."""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .config import PredictorConfig
from .risk_predictor import STRiskPredictor
from .scenarios import DynamicSAGINScenario, ScenarioConfig
from .topology import TimeVaryingTopology


@dataclass
class STGCNSample:
    node_input: torch.Tensor
    link_input: torch.Tensor
    adjacency: torch.Tensor
    endpoints: torch.Tensor
    node_target: torch.Tensor
    link_target: torch.Tensor
    trajectory_id: int
    time_index: int


def split_trajectory_ids(total_seeds: int, train_ratio: float, val_ratio: float, split_seed: int) -> Dict[str, List[int]]:
    if total_seeds < 3:
        raise ValueError("At least three trajectories are required for train/validation/test splitting.")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one.")
    trajectory_ids = list(range(total_seeds))
    random.Random(split_seed).shuffle(trajectory_ids)
    train_end = max(1, int(total_seeds * train_ratio))
    val_end = max(train_end + 1, int(total_seeds * (train_ratio + val_ratio)))
    if val_end >= total_seeds:
        val_end = total_seeds - 1
    return {
        "train": sorted(trajectory_ids[:train_end]),
        "val": sorted(trajectory_ids[train_end:val_end]),
        "test": sorted(trajectory_ids[val_end:]),
    }


def build_samples(
    scenario_config: ScenarioConfig,
    trajectory_ids: Sequence[int],
    slots: int,
    window: int,
    horizon: int,
    device: torch.device,
) -> List[STGCNSample]:
    if slots <= window + horizon:
        raise ValueError("slots must be greater than window + horizon.")
    label_predictor = STRiskPredictor(PredictorConfig(history_window=window, future_horizon=horizon))
    samples: List[STGCNSample] = []
    for trajectory_id in trajectory_ids:
        scenario = DynamicSAGINScenario(scenario_config, trajectory_id)
        snapshots = [scenario.snapshot(step) for step in range(slots)]
        for time_index in range(window - 1, slots - horizon):
            history = TimeVaryingTopology(snapshots[: time_index + 1])
            nodes, links, node_series, link_series = label_predictor._collect_series(history)
            adjacency, endpoints = label_predictor._graph_tensors(nodes, links, history.window(window), device)
            node_targets, link_targets = [], []
            for offset in range(1, horizon + 1):
                labels = label_predictor.ground_truth_risk(snapshots[time_index + offset])
                node_targets.append([labels.node_risk[node_id] for node_id in nodes])
                link_targets.append([labels.link_risk[link_id] for link_id in links])
            samples.append(
                STGCNSample(
                    torch.tensor(node_series, dtype=torch.float32, device=device).permute(1, 2, 0),
                    torch.tensor(link_series, dtype=torch.float32, device=device).permute(1, 2, 0),
                    adjacency,
                    endpoints,
                    torch.tensor(node_targets, dtype=torch.float32, device=device).transpose(0, 1),
                    torch.tensor(link_targets, dtype=torch.float32, device=device).transpose(0, 1),
                    trajectory_id,
                    time_index,
                )
            )
    return samples


def evaluate_predictions(
    model: torch.nn.Module,
    samples: Iterable[STGCNSample],
    horizon: int,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    values: Dict[str, List[float]] = {f"{kind}_{metric}_h{step}": [] for kind in ("node", "link") for metric in ("abs", "squared", "tp", "fp", "fn", "positive", "count") for step in range(1, horizon + 1)}
    with torch.no_grad():
        for sample in samples:
            node_prediction, link_prediction = model(sample.node_input, sample.link_input, sample.adjacency, sample.endpoints)
            for kind, prediction, target in (("node", node_prediction, sample.node_target), ("link", link_prediction, sample.link_target)):
                for index in range(horizon):
                    error = prediction[:, index] - target[:, index]
                    values[f"{kind}_abs_h{index + 1}"].extend(error.abs().cpu().tolist())
                    values[f"{kind}_squared_h{index + 1}"].extend(error.square().cpu().tolist())
                    predicted_positive = prediction[:, index] >= threshold
                    target_positive = target[:, index] >= threshold
                    values[f"{kind}_tp_h{index + 1}"].append(float((predicted_positive & target_positive).sum().item()))
                    values[f"{kind}_fp_h{index + 1}"].append(float((predicted_positive & ~target_positive).sum().item()))
                    values[f"{kind}_fn_h{index + 1}"].append(float((~predicted_positive & target_positive).sum().item()))
                    values[f"{kind}_positive_h{index + 1}"].append(float(target_positive.sum().item()))
                    values[f"{kind}_count_h{index + 1}"].append(float(target_positive.numel()))
    metrics: Dict[str, float] = {}
    for kind in ("node", "link"):
        for step in range(1, horizon + 1):
            mae = float(np.mean(values[f"{kind}_abs_h{step}"]))
            rmse = float(np.sqrt(np.mean(values[f"{kind}_squared_h{step}"])))
            tp, fp, fn = (sum(values[f"{kind}_{name}_h{step}"]) for name in ("tp", "fp", "fn"))
            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
            positive_count = sum(values[f"{kind}_positive_h{step}"])
            total_count = sum(values[f"{kind}_count_h{step}"])
            metrics.update({
                f"{kind}_mae_h{step}": mae,
                f"{kind}_rmse_h{step}": rmse,
                f"{kind}_precision_h{step}": precision,
                f"{kind}_recall_h{step}": recall,
                f"{kind}_f1_h{step}": f1,
                f"{kind}_positive_rate_h{step}": positive_count / max(1.0, total_count),
            })
    metrics["selection_loss"] = float(np.mean([
        metrics[f"node_rmse_h{step}"] ** 2 + metrics[f"link_rmse_h{step}"] ** 2
        for step in range(1, horizon + 1)
    ]))
    return metrics
