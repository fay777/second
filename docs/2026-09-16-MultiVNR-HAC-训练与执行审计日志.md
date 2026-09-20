# 2026-09-16：Multi-VNR HAC 训练与执行审计修改日志

## 今日目标

将冻结的 ST-GCN 与 Top-1 Selector 接入多 VNR 共享资源环境，形成可训练的分层 Actor-Critic（HAC）闭环，并定位训练初期重配置失败原因。

```text
原始物理 SAGIN 历史
    -> Frozen ST-GCN
    -> Frozen Top-1 Selector
    -> Upper HAC: KEEP / delay / scope
    -> Lower HAC: target physical hosts
    -> 确定性 risk-aware routing
    -> shared CPU/BW ledger transaction
```

ST-GCN、Selector 不参与反向更新，路由不作为 AC 动作。

## Multi-Service HAC 接入

- 新增 `pe_vnr/hac_worker.py` 与策略 `stgcn_topk_hac`。
- Worker 不推进时间、不递减生命周期、不提交共享资源账本；环境负责 `release -> execute -> commit/rollback -> reserve`。
- `stgcn_topk_hac` 默认绑定 `TopRiskKSelector()`，不会误用 `AllRiskySelector()`。
- Upper state 增加全局 residual CPU/BW pressure，反映共享资源竞争。
- `link-only` 只重路由；`partial/full` 使用 Lower HAC 选择目标宿主。

## 延迟决策与训练接口

每个 Upper action 有唯一 `decision_id`。pending 只保存：

```text
service_id, due_time, scope, created_at, decision_id
```

到期时使用最新风险预测和 residual graph 重新执行 Lower HAC 与 routing。`pending_execution` 用 `origin_decision_id` 将回报归因给原始 Upper action。

Worker 暴露 `upper_observation()`、`upper_decide()`、`execute_now()`；最后一个接口返回逐 VNF 的 `LowerTransition`。Trainer 负责采样、保存 `log_prob/value`、计算 return 和更新网络。

## 训练与 Reward 修正

新增 `train_multiservice_hac.py`，仅更新 Upper/Lower HAC，输出：

```text
upper.pt, lower.pt, training_history.json, metadata.json, execution_audit.json
```

### Lower credit

每次 partial/full execution 是独立 Lower 子轨迹。只有最后一个 placement action 得到 terminal reward，再由 discounted return 回传给此前 actions；不同服务 execution 不互相传播 return。

同时修复后续 VNF 无可行候选时，已采样 Lower action 未进入 trace 的问题：动作在请求下一个 observation 前记录，失败 execution 也能完整归因。

### SLA reward

- `violation -> normal`：正奖励。
- `normal -> violation`：强惩罚。
- `violation -> violation`：小惩罚。
- KEEP 根据后续 `H=3` 时隙的真实 SLA history 获取 `future_sla` penalty。

当前 reward 分量：`risk`、`cost`、`disruption`、`sla`、`future_sla`、`failure`。

## 执行审计

训练 history 记录 Upper 动作、scope、pending、Lower/routing 成败、rollback、成功率，以及每个 reward 分量的每 Upper action 均值。

`execution_audit.json` 对每次执行记录：

```text
episode, seed, time_step, service_id, scope, event_type
failure_stage, migrated, accepted, target_vnodes, candidate_counts
released_cpu_ratio, released_bandwidth_ratio, pre_risk, post_risk
```

failure stage 包含：

```text
accepted, unchanged, insufficient-risk-reduction
no_candidate_host, fixed_node_infeasible
no_feasible_route, preserved_path_infeasible
invalid_lower_action, lower_step_infeasible
```

## 执行可行性修正

### 路径可行候选筛选

Lower 候选宿主必须能够与已映射虚拟邻居通过当前 BW、fault、availability 和 visibility 约束建立物理可行路径。该机制仅是 candidate mask；实际路径仍由确定性 routing 选择。

### Scope feasibility

- 任一当前宿主在释放自身预留后仍不可行时，mask `link-only`。
- `partial` 的目标集包含高风险 VNF 与当前宿主不可行 VNF，避免保留不可行 fixed node。

### HAC 软风险约束

仅 `stgcn_topk_hac` 设置：

```text
enforce_risk_reduction = False
```

风险下降从 `Delta risk >= 0.01` 的硬拒绝条件变为 reward 软目标。CPU、BW、fault、visibility、候选宿主、routing 和 deployment changed 保持硬约束。每次 HAC 执行结束后恢复 Executor 配置，Reactive-Full 与 heuristic baselines 不受影响。

## 已验证结论

- 路径筛选显著降低 `no_feasible_route`。
- scope feasibility 修正显著降低 `fixed_node_infeasible`。
- 移除 HAC hard risk gate 后，`insufficient-risk-reduction` 归零，accepted migration 明显增加。
- `no_candidate_host` 是真实硬可行性约束，不能用 reward 或 gate 放宽掩盖。

以上均为 smoke 与训练诊断结果，不能作为论文最终性能结论。

## 下一步

执行框架现可冻结。下一阶段为 reward calibration：参数化六项 reward 权重，记录 accepted migration 的平均风险变化、成本、中断和 scope 分布；先跑 20--50 个训练 episode 观察经验尺度。reward 固定后才开始 train/validation/test seed 分离和最佳 checkpoint 选择。

## 2026-09-19 Reward Calibration 实现

`train_multiservice_hac.py` 现在将六项奖励拆为两层：

```text
raw reward：环境直接产生的信号
weighted reward：raw reward × 实验权重，实际用于 HAC policy update
```

可配置项为：

```text
--reward-risk-weight
--reward-cost-weight
--reward-disruption-weight
--reward-sla-weight
--reward-future-sla-weight
--reward-failure-weight
```

默认权重为 `risk=2`，其余均为 `1`，严格保持此前的 reward 数值语义。每次训练在 `metadata.json` 保存权重；`training_history.json` 同时保存 `reward_*`（加权）和 `reward_raw_*`（原始）分量，以及成功迁移的平均风险变化、实际成本、中断和 scope 分布。

此外，history 对每个 `link-only`、`partial`、`full` scope 保存：

```text
execution_scope_<scope>_attempts
accepted_scope_<scope>
accepted_rate_scope_<scope>
```

`reward_*_per_execution` 与 `reward_raw_*_per_execution` 仅在实际 execution 事件上计算，避免 KEEP 的 `future_sla` credit 混入 execution 分母。`future_sla_per_execution` 因而为零；该信号只用于衡量 Upper KEEP 的长期后果。

校准阶段先固定随机种子、训练长度、ST-GCN checkpoint 和 selector，仅改变一组 reward 权重。不要把 20--50 episode 的校准结果作为最终论文性能，也不要同时改变网络结构、K 或 routing。

### 首轮校准命令

先执行默认权重作为对照：

```bash
python -B train_multiservice_hac.py \
  --stgcn-checkpoint artifacts/stgcn_h3/stgcn_best.pt \
  --episodes 20 --steps 100 --seed 400 --diagnostic \
  --output artifacts/hac_calibration_default
```

然后仅提高风险项权重，保持其他条件完全一致：

```bash
python -B train_multiservice_hac.py \
  --stgcn-checkpoint artifacts/stgcn_h3/stgcn_best.pt \
  --episodes 20 --steps 100 --seed 400 --diagnostic \
  --reward-risk-weight 10 \
  --output artifacts/hac_calibration_risk10
```

首轮比较 `training_history.json` 中的 `reward_raw_*`、`reward_*`、`accepted_avg_risk_reduction`、`accepted_avg_realized_cost`、`accepted_avg_disruption` 与 `sla_violation_rate`。若风险项仍明显小于成本、中断、SLA 与失败项，再测试 `--reward-risk-weight 20`；不要一次同时更改多个权重。

## Actor-Critic 更新修正与第二轮校准

第一轮 `risk=2/10/20/40` 表明，线性放大风险权重无法改变 `risk=20` 与 `risk=40` 的采样行为。风险变化本身有区分度，且 accepted migration 的负风险变化幅度大于正风险变化幅度；因此不应改为 relative risk，也不应继续扫描更大的线性权重。

`update_policy()` 已修正为：

```text
critic target = raw discounted return
actor signal = normalize(return - detached value)
```

此前错误地标准化了 return 本身，导致 critic 学习的是每个 rollout 的标准化目标。第二轮的所有结果均不得与第一轮旧更新公式的 checkpoint 或性能表直接合并。

训练 history 新增 accepted migration 的：

```text
improved / worsened / unchanged count and fraction
mean positive / negative risk reduction
scope x risk reduction and improved/worsened/unchanged count
```

每个训练结束后，脚本会在固定的独立 calibration seeds `900, 901, 902` 上执行无采样的 argmax rollout，并写入 `calibration_evaluation.json`。这仅用于比较奖励设计，不属于正式 validation 或 untouched test。可传入 `--calibration-eval-seeds` 后不带数值以禁用。

## 节点/链路风险分量诊断

系统级风险继续使用 `max(mean_node_risk, mean_link_risk)`，因此与旧 baseline 和历史指标保持可比。HAC 的 execution outcome 另外记录：

```text
pre/post_node_risk, pre/post_link_risk
node_risk_reduction, link_risk_reduction
```

这些字段先用于诊断 `link-only` 重路由是否降低实际链路风险；若节点风险始终主导 `max(...)`，全局风险变化为零并不代表路由优化没有效果。诊断确认后再定义 HAC 专用风险 reward 组合，统一系统指标不变。

诊断结果确认：`link-only` 的 67 次 accepted reroute 中有 57 次降低链路风险，但全局与节点风险变化均为零；`partial` 的节点、链路风险平均均下降，`full` 的两类风险平均均恶化。因此 HAC reward 的 `risk` 分量已改为：

```text
node_risk_reduction + link_risk_reduction
```

该项不引入额外比例超参数，确保节点迁移和链路重路由都能获得与其实际风险变化一致的学习信号。统一系统指标继续使用 `max(mean_node_risk, mean_link_risk)`，所有 baseline 的统计定义不变。此前 raw-return 的 `risk=2/20/40` checkpoint 仅作诊断，不再作为此新 reward 定义的 calibration 对照。

## Upper 动作编码修正

原 PlanningEnv 使用 24 个编码，其中实际上只有一个合法 KEEP：

```text
KEEP = 1
MIGRATE(delay, scope) = 4 x 3 = 12
```

直接在 24 类 logits 上采样会使迁移拥有 12 倍初始概率质量，并使 argmax 偏向多个 migration logits 的最大值。最初将其压缩为 13 类动作仍不充分：单个 KEEP 类仍会与 12 个迁移类竞争总概率质量。因此 Upper HAC 改为真正的两阶段策略：

```text
stage 1: binary KEEP / MIGRATE
stage 2: only when MIGRATE, choose one of 4 delays x 3 scopes
```

网络输出为 14 个 logits：前 2 个用于二元决策，后 12 个仅在迁移条件下用于 `(delay, scope)`。执行前将条件动作映射回既有 PlanningEnv 的 `12..23` migration 编码。因此不需要修改 Worker、环境或规划 MDP 的语义，同时消除了动作空间基数导致的结构性迁移偏置。此前 component-reward 与 13 类 canonical-action 的所有 checkpoint 都只能作为诊断，不能与该两阶段策略的结果直接比较。

Argmax calibration evaluation 还记录二元决策的 `mean_keep_probability` 与 `mean_migration_vs_keep_logit_margin`。当 KEEP rate 为零时，这两个字段用于判断策略是接近 KEEP/MIGRATE 边界，还是迁移决策明显占优；在此诊断完成前不继续扫描风险权重。

## Reward 冻结与周期验证

在固定 calibration seeds `900, 901, 902` 的 deterministic argmax rollout 上，`risk_weight=20` 相对 `risk_weight=2` 取得稳定的正平均复合风险改善，同时未观察到 SLA 恶化。因此后续训练固定：

```text
risk=20
cost=disruption=sla=future_sla=failure=1
```

该结论只用于超参数冻结，不作为统计显著性结论。calibration seeds 自此封存，不参与 checkpoint 选择或最终测试。

长训练采用独立的 train / validation / test seed split。Trainer 支持每隔 `validation_interval` 个训练 episode，在 held-out validation seeds 上执行 deterministic argmax evaluation，并写出：

```text
validation_history.json
best_upper.pt
best_lower.pt
best_metadata.json
```

checkpoint 选择规则预先固定为：

```text
feasible <=> accepted_migrations > 0
            and accepted_avg_risk_reduction > 0
            and validation SLA < matched Static SLA

among feasible checkpoints:
minimize SLA violation
tie-break: total cost -> total disruption -> migrations
```

Trainer 会在 validation 开始前按相同 workload seeds 计算一次 Static SLA reference，并写入 `best_metadata.json`。要求正风险改善和严格优于 Static，避免全 KEEP 或极少量零风险迁移被误判为可行。成本和中断作为 tie-break 与监控项，不作为不稳定的硬可行性约束。若所有 checkpoint 均不满足约束，`best_checkpoint_found=false`，不产生 `best_upper.pt` / `best_lower.pt`。

建议的正式开发 split：训练 seed 从 `100` 起，validation 使用 `1200..1209`，最终 untouched test 使用 `2000..2019`。三者不得重叠。

## 常用命令

```bash
python -B train_multiservice_hac.py \
  --stgcn-checkpoint artifacts/stgcn_h3/stgcn_best.pt \
  --episodes 200 \
  --steps 100 \
  --seed 100 \
  --reward-risk-weight 20 \
  --calibration-eval-seeds \
  --validation-seeds 1200 1201 1202 1203 1204 1205 1206 1207 1208 1209 \
  --validation-interval 10 \
  --output artifacts/hac_train_risk20
```

```bash
python -B smoke_hac_worker.py
```
