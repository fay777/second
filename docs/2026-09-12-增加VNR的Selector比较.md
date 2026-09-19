# 2026-09-12：增加 VNR 的 Selector 比较

## 目标

在 100 节点、多运行 VNR 的 SAGIN 环境中加入非 RL 服务选择层，形成：

```text
Raw SAGIN history
    -> Frozen ST-GCN
    -> Risk-aware service selector
    -> Heuristic / HAC reconfiguration worker
    -> Shared residual-resource graph
```

Selector 不作为第三层强化学习模块，而是以受控重配置预算避免同一时隙多个业务同时迁移。

## 1. 新增服务选择器

新增 `pe_vnr/service_selector.py`：

- `AllRiskySelector`：选择所有超过风险阈值且没有 pending plan 的运行服务。
- `TopRiskKSelector`：仅选择风险分数最高的 K 个候选服务，默认 `K=1`。
- `ServiceSelectionRecord`：记录服务峰值风险、期望风险、优先级、分数、候选排序、选择状态与 pending 状态。

第一版服务优先级分数固定为：

```text
Score_i = P_i * [0.7 * R_i_peak + 0.3 * R_i_exp]
```

其中 `R_i_peak` 为业务当前节点/链路映射在未来 H 个时隙中的最大风险，`R_i_exp` 为统一多步期望风险。低于风险阈值或已有 pending migration 的服务不会进入候选集。

暂不加入 SLA margin、迁移成本、CPU/BW 利用率等额外人工权重，也不引入 Selector RL。

## 2. 统一风险语义

Selector 的期望风险改为调用 `RiskPrediction.expected_risk_maps()`，不再自行按 horizon 重新聚合。这样 Selector、Planner、Executor 对多步期望风险使用同一套定义；峰值风险仅用于识别高风险服务和排序。

## 3. 分离物理拓扑与共享资源账本

`MultiServiceDynamicEnv` 改为维护两个对象：

- `physical_topology`：原始物理 SAGIN 快照历史，仅供冻结 ST-GCN 推理。
- `residual_topology`：扣除运行 VNR CPU/BW 预留后的当前资源图，仅供准入、迁移执行与风险感知路由。

预测调用改为：

```python
prediction = self.predictor.predict(self.physical_topology)
```

此修改避免将运行 VNR 的资源占用错误解释为底层拓扑容量退化，同时使 ST-GCN 在线输入与其原始物理轨迹训练数据保持一致。

## 4. 串行执行和 pending plan

每个时隙中：

1. ST-GCN 只对全局物理拓扑预测一次。
2. Selector 从全体运行服务提取风险暴露并进行排序。
3. 已到期的 pending plan 优先执行。
4. 新选中的 Top-K 服务随后逐个进入规划和执行模块。
5. 每完成一个服务的重配置，立即更新共享 CPU/BW 账本，再处理下一个服务。

具有 pending plan 的服务不会被重复提交给 Selector，从而避免重复安排迁移。延迟迁移在到期时才读取最新剩余资源状态执行。

## 5. 基线语义修正

恢复旧 `proactive_heuristic` 的定义：

```text
Heuristic predictor + all active services enter heuristic planner
```

它不使用 Selector。

新增显式策略：

```text
heuristic_all_risky
    = Heuristic predictor + All-Risky selector + Heuristic planner

stgcn_all_risky
    = Frozen ST-GCN + All-Risky selector + Heuristic planner

stgcn_topk_heuristic
    = Frozen ST-GCN + Top-Risk-K selector + Heuristic planner
```

旧 Pilot-1 中预测器读取 residual topology，属于已修正的错误语义；旧结果仅保留为环境调试记录，不进入正式论文表格。

## 6. 指标调整

Selector 相关指标改为明确命名：

- `selector_candidate_count`
- `selector_selected_count`
- `selector_pending_count`
- `avg_active_pending_plans`
- `max_active_pending_plans`

其中前 3 项属于 Selector 层统计；pending 队列规模独立统计，避免混淆“未被选择”和“系统中不存在延迟迁移”。

同时在 `MetricsTracker` 中新增系统总量指标：

- `total_realized_cost`
- `total_disruption`
- `total_node_migration_cost`
- `total_link_reroute_cost`

不能仅用单次平均成本比较迁移次数相差很大的策略。

## 7. 验证

完成以下检查：

- `TopRiskKSelector(top_k=1)` 仅选择优先级加权分数最高的候选服务。
- `physical_topology` 与 `residual_topology` 为独立对象。
- 旧 `proactive_heuristic` 的 Selector 统计为零，未被新 Selector 语义污染。
- 单策略 smoke 能输出 `selection_metrics.csv`，包含服务风险、优先级、排序和 pending 信息。
- 多服务 baseline runner 支持 checkpoint、策略组合与 `--selector-top-k` 参数。

## 8. Selector Pilot 结果

在 10 seeds、100 slots、`K=1` 的初步比较中：

| 策略 | SLA 违规率 | 可用性 | 平均迁移次数 |
| --- | ---: | ---: | ---: |
| Static | 0.5897 | 0.9979 | 0.0 |
| Reactive-Full | 0.4581 | 0.9917 | 180.8 |
| Proactive-Heuristic | 0.5981 | 0.9230 | 176.3 |
| STGCN-AllRisky | 0.5866 | 0.9979 | 1.6 |
| STGCN-TopK | 0.5897 | 0.9979 | 0.6 |

解释：

- Heuristic predictor 造成大量无效主动迁移，风险下降有限且显著损害可用性。
- Frozen ST-GCN 显著减少误触发，`STGCN-AllRisky` 的可用性恢复到 Static 水平。
- Top-K Selector 能限制决策预算：约 202 个候选服务中仅约 44 个进入 Top-1 决策，但启发式 Upper Planner 对绝大多数候选输出 `KEEP`，最终平均只迁移 0.6 次。
- 因此当前 Top-K 没有显著改变 SLA，不应据此宣称其已经优于 All-Risky；瓶颈在启发式规划器，而非 Selector 选择逻辑。

## 9. 后续工作

不继续通过调阈值或直接增加 K 强行放大启发式迁移。下一步接入现有 Upper/Lower HAC，并在多 VNR 共享资源环境重新训练 HAC：

```text
Frozen ST-GCN + Top-1 Selector
    -> Upper HAC: keep / delay / scope
    -> Lower HAC: target physical node
    -> Risk-aware routing
    -> Serial residual-ledger update
```

冻结 ST-GCN 与 Selector 架构；保留 HAC 网络结构，但不复用单服务训练权重作为最终论文模型。
