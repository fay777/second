# 2026-09-12：增加 VNR 的 Selector

## 目标

在多 VNR 共享 CPU/BW 账本中加入非 RL 的风险感知服务调度层。该层只决定本时隙哪些运行服务进入重配置决策，不替代分层 Actor-Critic 的职责：

```text
Raw SAGIN history
    -> Frozen ST-GCN
    -> Risk-aware service selector
    -> Upper HAC: whether / when / scope
    -> Lower HAC: where
    -> Risk-aware routing
    -> Residual shared-resource graph
```

Selector 不是第三个强化学习模块。其作用是在多个高风险运行服务竞争资源时施加重配置预算，避免同一时隙所有服务同时迁移。

## 新增模块

新增 `pe_vnr/service_selector.py`。

- `AllRiskySelector`：选择全部满足风险阈值且没有 pending plan 的服务。
- `TopRiskKSelector`：只选择评分最高的 K 个候选服务。
- `ServiceSelectionRecord`：记录每个服务的峰值风险、期望风险、优先级、分数、排序、是否选中及 pending 状态。

评分函数固定为：

$$
Score_i=P_i[\alpha R_i^{peak}+(1-\alpha)R_i^{exp}],
$$

其中默认 $\alpha=0.7$。候选服务必须满足 $R_i^{peak}\geq\tau_s$，且没有已安排的 pending migration。第一版不加入 SLA margin、迁移成本、CPU/BW 等额外手工权重。

风险定义：

- $R_i^{peak}$：服务当前节点/链路映射所涉及物理资源在未来 H 步预测中的最大风险。
- $R_i^{exp}$：通过 `RiskPrediction.expected_risk_maps()` 聚合后的期望风险，再从该服务映射资源中提取。该定义与 Planner、Executor 使用的期望风险保持一致。

## 关键语义修复

### 1. 分离物理拓扑与剩余资源图

`MultiServiceDynamicEnv` 现在维护两个对象：

- `physical_topology`：未扣除运行 VNR 预留资源的原始物理 SAGIN 历史，只供 ST-GCN 推理。
- `residual_topology`：扣除当前运行服务 CPU/BW 占用后的剩余资源图，只供准入、执行和路由可行性使用。

预测调用固定为：

```python
prediction = self.predictor.predict(self.physical_topology)
```

原因是 ST-GCN 的训练数据来自 `scenario.snapshot(t)` 的原始物理轨迹。若把业务预留后的 residual BW/CPU 输入冻结预测器，会把业务负载误判为底层物理风险，造成训练/推理分布不一致。

### 2. 保留旧 Proactive-Heuristic 基线

`proactive_heuristic` 保持历史定义：

```text
Heuristic predictor + all active services -> heuristic planner
```

它不经过 Selector。新增独立策略：

```text
heuristic_all_risky
    = Heuristic predictor + All-Risky selector + heuristic planner

stgcn_all_risky
    = Frozen ST-GCN + All-Risky selector + heuristic planner

stgcn_topk_heuristic
    = Frozen ST-GCN + Top-Risk-K selector + heuristic planner
```

旧 Pilot-1 中预测器读取 residual topology，属于错误输入语义；该 Pilot 仅作为环境调试记录，不能进入论文正式主表。今后所有正式 baseline 都必须使用 `physical_topology` 重新运行。

### 3. pending plan 的执行顺序

Selector 会排除已有 pending plan 的服务，避免同一服务重复安排延迟迁移。到期计划仍优先执行：

```text
due pending services
    -> execute with latest residual resources
    -> newly selected Top-K services
    -> update ledger after each service
```

因此上层输出 `(migrate=1, delay=d, scope)` 时，Lower worker 不会提前占用未来资源；到期后才在最新共享资源状态中执行。

## 指标调整

Selector 相关字段统一命名，避免旧 `proactive_heuristic` 被误解为没有进入 Planner：

```text
selector_candidate_count
selector_selected_count
selector_pending_count
avg_active_pending_plans
max_active_pending_plans
```

前三项只描述 Selector 层行为。`proactive_heuristic` 不使用 Selector，因此这些指标为零；它仍会将全部活跃服务送入 Planner。

同时在 `pe_vnr/metrics.py` 增加系统总量指标：

```text
total_realized_cost
total_disruption
total_node_migration_cost
total_link_reroute_cost
```

不能只比较 `avg_realized_cost` 或 `avg_disruption`，因为它们只对重配置记录求均值；例如 180 次迁移与 0.6 次迁移的方法必须比较总迁移成本和总中断。

## 已完成验证

- `TopRiskKSelector(top_k=1)` 单元验证：只保留优先级加权风险分数最高的服务。
- 多服务运行 smoke：Selector 日志成功输出 `selection_metrics.csv`，包含 `peak_risk`、`expected_risk`、`priority`、`score`、`rank`、`selected`、`has_pending_plan`。
- 物理图与剩余资源图分离断言通过。
- 旧 `Proactive-Heuristic` 与显式 `Heuristic-AllRisky` 的短运行均通过；旧基线的 Selector 统计保持为零。

## Selector Pilot 结果与解释

在 10 seeds、100 时隙的正确拓扑语义下：

| 方法 | 平均迁移次数 | SLA 违规率 | 可用性 |
| --- | ---: | ---: | ---: |
| Static | 0.0 | 0.5897 | 0.9979 |
| Reactive-Full | 180.8 | 0.4581 | 0.9917 |
| Proactive-Heuristic | 176.3 | 0.5981 | 0.9230 |
| STGCN-AllRisky | 1.6 | 0.5866 | 0.9979 |
| STGCN-Top1 | 0.6 | 0.5897 | 0.9979 |

诊断结论：

- Heuristic predictor 导致大量低收益迁移，造成可用性下降。
- ST-GCN 将风险候选交给启发式 Planner 后，大多数候选得到 `KEEP`，因而显著减少迁移。
- Top-1 已正常选出服务，但启发式 Upper Planner 仍几乎总是 `KEEP`；所以 Top-K 的收益暂时无法由启发式 worker 充分体现。
- pending plan 计数为零，说明当前启发式规划几乎没有选择延迟迁移，`when to migrate` 需要由后续 HAC 学习。

当前不要直接比较 K=2、K=3，也不要为增加迁移次数而调 heuristic 阈值。此时的核心瓶颈是启发式规划器，不是 Selector。

## AutoDL 同步文件

```text
pe_vnr/service_selector.py
pe_vnr/multi_service_env.py
pe_vnr/metrics.py
run_multiservice.py
run_multiservice_baselines.py
```

## 重新运行正式 Selector Pilot

```bash
python run_multiservice_baselines.py \
  --seeds 10 \
  --steps 100 \
  --arrival-probability 0.45 \
  --max-active-services 20 \
  --policies static,reactive_full,proactive_heuristic,stgcn_all_risky,stgcn_topk_heuristic \
  --stgcn-checkpoint artifacts/stgcn_h3/stgcn_best.pt \
  --device cuda \
  --selector-top-k 1 \
  --output-dir artifacts/multiservice_selector_top1
```

该命令需在同步新增总成本指标后重新运行，正式表格报告 mean +/- std，并加入总迁移成本与总中断。

## 后续衔接

下一阶段接入现有分层 HAC，但不增加 Selector RL：

```text
Frozen ST-GCN + Top-1 Selector
    -> multi-service Upper HAC worker
    -> multi-service Lower HAC worker
    -> shared residual ledger update
    -> retrain planning/execution HAC only
```

ST-GCN、Selector 公式及 Top-K 机制可以冻结；旧单服务 HAC 权重不能冻结，必须在多 VNR 共享资源环境重新训练或微调。
