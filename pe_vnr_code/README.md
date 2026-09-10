# Proactive Elastic VNR Code

This project implements a standalone codebase for the paper:

`Proactive Elastic Virtual Network Reconfiguration via Topology Prediction for Dynamic Space-Air-Ground Integrated Networks`

The pipeline follows the paper design:

1. Running virtual network
2. Historical topology sequence
3. ST-GCN-like future risk prediction
4. Actor-Critic migration planner
5. Risk-aware reconfiguration executor
6. New stable deployment

## Structure

```text
pe_vnr_code/
├── main.py
├── run_baselines.py
├── train_hierarchical.py
├── requirements.txt
├── README.md
└── pe_vnr/
    ├── __init__.py
    ├── config.py
    ├── topology.py
    ├── risk_predictor.py
    ├── planner.py
    ├── executor.py
    ├── demo.py
    └── mdp/
        ├── __init__.py
        ├── actor_critic.py
        ├── planning_env.py
        └── execution_env.py
```

## What is implemented

- Standalone time-varying topology and service data model
- ST-GCN-like temporal risk predictor with a heuristic fallback
- Migration planning layer
- Elastic reconfiguration execution layer
- Two MDP skeletons for planning and execution
- A runnable demo that goes from topology history to stable redeployment
- Matched-trajectory baselines and a masked two-policy A2C training entry point

## Quick start

```bash
python main.py
```

## Baselines

Run the static, reactive, and proactive-heuristic policies on the same seeded
topology trajectories:

```bash
python run_baselines.py --seeds 10 --steps 6 --output results
```

The script writes `baseline_runs.csv` (per seed) and `baseline_summary.csv`
(mean and standard deviation) for later plotting and significance testing.

## Hierarchical Training

The training environment is staged: `planning_step(action)` returns either the
next planning state or an execution state; `execution_step(action)` places one
virtual node and finishes the reconfiguration when all targets are placed.

```bash
python train_hierarchical.py --episodes 100 --output checkpoints
```

The supplied trainer is a compact masked A2C reference implementation. Its
risk predictor uses the deterministic heuristic mode. Train and validate an
ST-GCN predictor separately before reporting learned-policy results.

## Notes

- This project is independent from `hrl-acra-main`.
- It reuses only the design idea of graph-aware actor-critic decomposition.
- The current implementation emphasizes paper-code structure and runnable flow.
- `PlanningEnv` masks invalid no-migration action combinations.
- `ExecutionEnv` masks infeasible physical-node actions at each placement step.
