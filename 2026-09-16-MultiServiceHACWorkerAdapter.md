# 2026-09-16：Multi-VNR HAC Worker Adapter

## 实现目标

新增 `pe_vnr/hac_worker.py`，将单 VNR Upper/Lower HAC 的决策逻辑接入多 VNR 共享资源环境，同时保持：

```text
一个全局时钟
一个共享 CPU/BW ledger
每个时隙一次 ST-GCN 推理
```

Worker 不推进时间、不递减生命周期、不提交资源账本，也不修改 `RunningService`。

## 当前流程

```text
Frozen ST-GCN
    -> Top-1 Selector
    -> MultiServiceHACWorkerAdapter.upper_decide
    -> KEEP / delay / scope
    -> immediate 或 due-time execute_now
    -> MultiServiceDynamicEnv release -> validate -> commit/rollback
```

`link-only` 不调用 Lower HAC；仅以 risk-aware routing 重路由受影响虚链路。`partial` 在真正执行时依据当前风险图重新定位目标 VNF 和关联虚链路。延迟决策只保存：

```text
service_id
due_time
scope
created_at
decision_id
```

不会保存旧目标宿主或旧链路路径。

## 新 policy

`MultiServiceDynamicEnv` 新增：

```text
stgcn_topk_hac
```

该 policy 需要显式传入 `MultiServiceHACWorkerAdapter`。它不会经过 `MigrationPlanner.plan()`；Selector 选中的服务直接进入 Upper HAC。

未显式传入 Selector 时，`stgcn_topk_hac` 默认构造 `TopRiskKSelector()`，不会误退化为 `AllRiskySelector`。

## 训练接口

Worker 提供：

```python
upper_observation(service, residual_graph, prediction, physical_graph)
upper_decide(..., action=sampled_upper_action)
execute_now(..., lower_action_fn=sample_and_record_lower_action)
```

`upper_observation` 的末两维为全局 residual CPU/BW pressure：当前残余资源相对当前原始物理可用资源的比例。它使 Upper HAC 能区分“同一服务、但共享资源宽松”与“同一服务、但共享资源紧张”。

`execute_now` 返回逐 VNF 的 `LowerTransition` trace。`MultiServiceDynamicEnv.run()` 额外返回 `hac_events`，其中保留 Upper decision、可选执行结果和 Lower trace。每个 Upper action 都有唯一 `decision_id`；延迟执行事件通过 `origin_decision_id` 回溯到创建该 pending decision 的 Upper action，不能错误归因给到期时的其他选择。

## 训练器

新增 `train_multiservice_hac.py`。它不修改环境、Worker、ST-GCN 或 Selector，只完成：

```text
sample Upper/Lower action
-> collect log_prob/value
-> run a multi-VNR rollout
-> use HAC event trace assign credit
-> update Upper/Lower actor-critic
```

ST-GCN checkpoint 以 `eval()` / `no_grad()` 方式被冻结，Selector 固定为默认 Top-1。每回合输出 `training_history.json`，保留 SLA、availability、admission、迁移数、累计成本、中断和六项 reward component：risk、cost、disruption、sla、future_sla、failure。模型 metadata 固定记录 ST-GCN checkpoint、网络维度、Top-1 和 reward 定义。

Lower 不再把一次 execution 的终端 reward 重复分配给每个 VNF。一次 partial/full 重配置中的 Lower actions 被单独视为一个子轨迹，只有最后一步得到终端 reward，再由 discounted return 回传给此前的 placement actions；不同服务的 Lower execution 不会互相传播 return。

HAC execution event 还记录 `pre_sla_violated` 和 `post_sla_violated`。迁移的 SLA reward 依据状态变化给出：`violation -> normal` 奖励，`normal -> violation` 惩罚，`violation -> violation` 给予较小惩罚。对于 KEEP，Trainer 利用环境记录的后续 3 个时隙服务 SLA history 施加 `future_sla` penalty，使主动决策不会只看当前时隙。

例如：

```bash
python train_multiservice_hac.py --stgcn-checkpoint artifacts/stgcn_h3/stgcn_best.pt --episodes 100 --steps 100 --output artifacts/multiservice_hac
```

## 事务语义

环境执行 HAC 时严格遵循：

```text
release old reservation
-> Worker decides on released residual graph copy
-> validate candidate deployment
-> commit new deployment or retain old deployment
-> reserve committed deployment
```

Worker 失败、路由失败或候选方案未通过风险下降约束时，环境重新预留旧部署，并累计 `rollback_count`。

## 新诊断指标

环境返回：

```text
upper_keep_count
upper_immediate_count
upper_delayed_count
scope_link_only_count
scope_partial_count
scope_full_count
pending_created
pending_executed
pending_cancelled
lower_decisions
lower_success
lower_failure
routing_success
routing_failure
rollback_count
```

## 验证

```bash
python smoke_hac_worker.py
```

已覆盖：KEEP 无副作用、immediate full、delayed partial 到期重新定位、Lower/routing 不可行时的外部图回滚。

短训练 smoke 已验证 50 时隙 rollout 能采集 Upper/Lower transitions、按 execution 分割 Lower 子轨迹、完成梯度更新并写出模型和日志。该 smoke 不构成实验结论；正式训练仍必须划分独立 train/validation/test seeds，并以 validation 选择 checkpoint。
