# 2026-09-16：Multi-VNR HAC 执行与奖励诊断

## 范围

本文记录 MultiService HAC Worker Adapter 之后完成的实现与诊断工作。ST-GCN 保持冻结，Top-1 Selector 保持不变。

## 训练与回报归因

- 新增 `train_multiservice_hac.py`，用于多 VNR Upper/Lower HAC 训练。
- 新增 `decision_id`，将延迟执行结果归因到原始 Upper action。
- 新增 `upper_decision` 与 `pending_execution` 事件记录。
- 每次执行的 Lower 决策是独立子轨迹；仅最后一个 Lower action 获得终端回报，return 不跨服务传播。
- 执行回报采用迁移前后 SLA 状态变化；KEEP 在观测到的三时隙 ST-GCN horizon 内接收 future-SLA penalty。

## 诊断

- 新增 reward 分量：`risk`、`cost`、`disruption`、`sla`、`future_sla`、`failure`，以及每个 Upper action 的归一化数值。
- 新增 `--diagnostic`，输出 Upper 动作、scope、pending 执行、Lower/routing 结果、rollback 和回报分量。
- 新增 `execution_audit.json`，记录时隙、服务、scope、失败阶段、目标 VNF 数、候选数、释放后的 CPU/BW 比例和风险值。

## 执行可行性修正

- 新增失败阶段：`no_candidate_host`、`fixed_node_infeasible`、`preserved_path_infeasible`、`no_feasible_route`、`unchanged`、`insufficient-risk-reduction` 和 `accepted`。
- 候选筛选要求在当前带宽、可见性和故障状态下，候选宿主可与每个已映射虚拟邻居建立物理可行路径。路由仍是确定性过程，不是 AC 动作。
- 当当前 VNF 宿主在释放自身预留后不可行时，mask `link-only`。
- partial 执行必须包含所有当前宿主不可行的 VNF。

## HAC 软风险约束

仅对 HAC，风险下降是 reward 软目标，不再是硬 acceptance gate。CPU、BW、fault、visibility、placement、routing 和 deployment changed 仍是硬约束。每次 HAC 事务后恢复 Executor 配置，保留 Reactive-Full 与 heuristic baseline 的原有语义。

## 当前状态

```text
冻结 ST-GCN                         已完成
冻结 Top-1 Selector                  已完成
Multi-VNR HAC Worker                  已完成
延迟决策回报归因                      已完成
执行失败审计                           已完成
路径与 scope 可行性修正                已完成
仅 HAC 的软风险约束                    已完成
Reward calibration                    下一阶段
Validation checkpoint selection       待开始
未参与调参的最终测试                   待开始
```

## 下一步

1. 冻结当前执行框架。
2. 在不改变环境语义的前提下参数化 reward 权重。
3. 运行 20--50 个训练 episode，检查 reward 尺度、accepted migration 风险变化、SLA 趋势和 scope 分布。
4. 固定 reward 权重后，再进行 train/validation/test seed 划分。
