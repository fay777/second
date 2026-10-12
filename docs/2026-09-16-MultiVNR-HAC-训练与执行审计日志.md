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

长训练进一步发现，仅对迁移结果奖励风险下降会导致 Upper 选择全 KEEP：迁移承担成本、中断与失败惩罚，而主动迁移若避免未来违约却没有获得长期 credit。曾尝试对高风险 KEEP 施加预测风险暴露惩罚，但该信号在实际候选上过于稀疏，仍会塌缩。

现在改为所有 Upper action 都依据原始决策时刻之后 `H=3` 个时隙的真实 SLA history 接收 `future_sla` credit：

```text
r_future_sla = - mean(SLA violations over the next H slots)
```

因此 KEEP 导致后续违约仍被惩罚；而即时或延迟迁移若避免后续违约，则不再遗漏其长期收益。迁移仍获得实际的 `node_risk_reduction + link_risk_reduction`，且不新增 reward 权重。此前 KEEP-risk-exposure 版本仅为失败诊断，不参与后续实验。

当前先不引入任意的 KEEP-risk 系数。Trainer 仅记录所有 Upper decision 的预测风险暴露：

```text
component exposure = expected node risk + expected link risk
peak horizon risk = max(predicted deployment risks over H horizons)
```

训练与 validation 分别输出 KEEP / MIGRATE 的两类 exposure，用于后续确定是否确有“高风险 KEEP”且需要额外 penalty。这样不会在缺少尺度证据时人为引入新的 reward 超参数。

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

在早期固定 calibration seeds `900, 901, 902` 的 deterministic argmax rollout 上，`risk_weight=20` 相对 `risk_weight=2` 表现出更强的正复合风险改善信号。随后，component-risk 与 future-SLA 的归因语义更新，因此不能将旧结论表述为最终最优；目前仅保留 `risk_weight=20` 作为唯一候选，不继续扫描 `40` 或更大的权重。

```text
risk=20
cost=disruption=sla=future_sla=failure=1
```

该结论不构成统计显著性结论。calibration seeds 自此封存，不参与 checkpoint 选择或最终测试。

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
            and accepted_avg_component_risk_reduction > 0
            and validation SLA < matched Static SLA

among feasible checkpoints:
minimize SLA violation
tie-break: total cost -> total disruption -> migrations
```

Trainer 会在 validation 开始前按相同 workload seeds 计算一次 Static SLA reference，并写入 `best_metadata.json`。其中 `component risk = node risk reduction + link risk reduction` 与 HAC 的训练 reward 完全一致；`global risk = max(mean node risk, mean link risk)` 继续作为论文统一报告指标。要求正 component risk 改善和严格优于 Static，避免全 KEEP 或极少量零风险迁移被误判为可行。成本和中断作为 tie-break 与监控项，不作为不稳定的硬可行性约束。若所有 checkpoint 均不满足约束，`best_checkpoint_found=false`，不产生 `best_upper.pt` / `best_lower.pt`。

建议的正式开发 split：训练 seed 从 `100` 起，validation 使用 `1200..1209`，最终 untouched test 使用 `2000..2019`。三者不得重叠。

## Batch Rollout 稳定性诊断

当使用单 episode 更新时，连续 validation 曾出现“第 10 回合满足 SLA/风险约束、而第 20 回合 SLA 回退”的策略漂移。为检验是否由高方差 on-policy 更新造成，Trainer 增加：

```text
--update-batch-episodes N
```

每个 episode 是一个独立的采样单元。Upper 在 episode 内按 `service_id` 拆成多条 trajectory，discounted return 只沿同一服务的决策累计，既不跨服务，也不跨 episode。Lower 仍按每次 execution 分成独立 trajectory。收集 `N` 个 episode 后，对所有 records 统一标准化 advantage 并执行一次 Upper/Lower optimizer update。validation interval 必须能被 `N` 整除，以保证评估的是刚更新的策略。

第一轮稳定性诊断固定为：

```text
episodes = 50
update_batch_episodes = 5
validation_interval = 10
```

即进行 10 次优化更新、5 次 held-out validation。训练 history 在 batch 结束行记录 actor loss、critic loss、raw-return mean/std、raw-advantage std 与 actor/critic 各自的裁剪前 gradient norm；非 batch 结束行的这些字段为 `null`。

Upper 和 Lower 的 actor/critic 均为独立 MLP，不共享 trunk。由于 critic 对 raw discounted return 的 MSE 可远大于 policy loss，不能对整个模型执行一次全局梯度裁剪，否则 critic 的大梯度会连带缩小 actor 梯度。Trainer 因此分别以 `1.0` 裁剪 actor 与 critic 参数组；这不改变 reward、学习率、熵正则、Selector、ST-GCN 或路由。

若 batch rollout 后仍出现 KEEP 塌缩，先运行 critic-fit diagnostic，而不是直接修改 reward 或 value target。每个 optimizer update 记录：

```text
return mean/std
value mean/std
critic RMSE
Pearson corr(value, return)
explained variance = 1 - Var(return - value) / Var(return)
```

同时记录 Upper 二元 gate 的 KEEP/MIGRATE 数量、即时 reward 平均值和 discounted-return 平均值。若 EV 持续接近或低于零，且 value 未跟随 return 的均值与尺度，再单独测试固定比例 return scaling；若 KEEP 的实际 return 系统性更高，则优先检查 reward/跨服务 credit assignment，而不是误判为 critic 数值问题。

固定比例 scaling 由 `--critic-return-scale c` 控制。critic target 使用 `cG`，actor 使用 `cG-V(s)` 后仍统一标准化 advantage；因此不会改变 risk、cost、SLA 等 reward 分量的相对偏好。诊断日志会将 value、RMSE 和 return 统一反变换到原始 reward 单位。第一轮仅测试 `c=0.05`，不同时修改学习率、reward 或 credit assignment。

本地 `c=0.05` 的 20 回合诊断中，Upper EV 仍为负，约 `-0.026` 至 `-0.036`；固定 scaling 未改善 critic。下一轮恢复默认 `c=1.0`，仅测试按服务分组的 Upper trajectory。执行事件仍按 `origin_decision_id` 将即时或延迟奖励归给原 Upper 决策。由于服务竞争共享 CPU/BW，按服务分组也会忽略部分跨服务外部性，因此此改动先作为 credit assignment 消融诊断，根据 critic EV、KEEP 比例与 held-out SLA 判断是否保留。

本地按服务分组的 20 回合诊断（`risk=20`、`c=1.0`、batch=5、seed=100）共完成四次更新。每批约 50 条服务轨迹，Upper return 均值约 `-3.6` 至 `-4.8`，RMSE 约 `5.3` 至 `7.5`；相比旧跨服务串联目标，目标尺度显著降低。但 EV 仍为 `-0.006` 至 `-0.003`，value 均值约 `-0.03` 至 `0.02`，critic 仍未形成有效 baseline。KEEP 的平均即时回报和 discounted return 均高于 MIGRATE。此次运行禁用 validation，只验证 credit assignment 和训练接口，不能据此判断系统 SLA 已改善。下一步应针对 critic 的学习速度与回报归因分别设计对照，不要把 RMSE 的下降误写成预测能力提高。

## 2026-10-11：Critic 学习速度与 Upper 回报归因修正

本轮检查发现，旧版 Upper critic 的 value 均值接近 0，而 raw return 明显为负；batch 之间的 pre-update explained variance 长期约为 0。仅缩放 critic target 没有解决拟合问题。本轮将 actor、critic 使用独立 optimizer，新增 `--critic-lr` 和 `--critic-updates`；actor 每批更新一次，critic 可在同一批状态及原始 discounted return 上做多次拟合。日志分别记录更新前、后的 RMSE/EV。更新后的同批 EV 只能证明拟合，不等于泛化；跨批次 pre-update EV 更重要。

Upper return 现在只沿同一 service 的决策链累计，并按真实间隔 `gamma ** elapsed_slots` 折扣。延迟执行产生的 reward 按相同时间间隔归给原始 decision；同一服务的未来 SLA 观察窗口截止到下一次 Upper 决策，避免重复归因。KEEP 在当前已违反 SLA 时也支付当前 SLA penalty；此前这一项只落到 MIGRATE execution，造成不公平的 gate 比较。日志加入 KEEP/MIGRATE 分组的已归因 reward components。此分组均值是观察性统计，不是同一状态下的因果反事实。

本地 20-episode 诊断采用 `risk=20`、batch=5、`critic-lr=0.003`、`critic-updates=20`，Upper 各批更新前 EV 为 `-0.004、+0.167、+0.138、+0.294`，说明 critic 开始学习；但固定 argmax 开发评估在 `1200..1202` 三个 seed 上仍为 100% KEEP，SLA 与 Static 相同。不能把 critic 改善解释为最终策略成功。训练执行审计有 188 次接受、195 次无候选宿主、35 次无可行路由。无候选中 125 次发生在首个 Lower 动作前，70 次发生在部分 VNF 放置之后；日志现分别记录 `no_candidate_initial` 与 `no_candidate_after_placement`。后续应针对真正不可行与顺序放置死路做对照，而不是直接放宽约束、调 reward 或宣称已经优于 Static。

这轮诊断可复现的参数组合是 `--critic-lr 0.003 --critic-updates 20 --update-batch-episodes 5 --reward-risk-weight 20`。尚不建议启动正式 200-episode 长训练。

随后针对无候选宿主，将 ExecutionEnv 的 candidate mask 改为轻量的剩余 CPU-host 二分图匹配必要条件：某个当前候选若会占用后续 VNF 的唯一 CPU 可行宿主，就从 Lower action mask 移除。该检查不估计路由，也不放宽资源限制；路由仍在 placement 完成后确定性执行。由于全量“候选 x 路由 x 匹配”前瞻对 100 节点场景计算代价过高，未采用该实现。

Upper action mask 也新增即时 remap 启动检查。对 `delay=0` 的 `partial/full`，若当前 released residual graph 中没有任何可启动的 Lower placement，则将该 action mask；对 `delay>0` 保持可选，到期时仍基于最新 residual graph 重新判断。这样不会把未来可能恢复可行的延迟迁移提前禁止。60-slot 本地 smoke 正常完成 24 次 Upper 决策、12 次 execution、7 次 accepted migration，记录 `no_candidate_initial=0`、`no_candidate_after_placement=1`。这是接口验证，不构成与旧版本的性能对比，也不证明 risk/SLA 已改善。

启动检查随后修正为先在 Worker 私有副本中释放当前 service 的 CPU/BW reservation，再构造 partial/full plan 和候选 mask；正式环境的执行事务也是先 release 再 remap。旧实现错误地用“尚包含当前服务预留”的 graph 预检查，可能把可通过交换原宿主完成的 full remap 错误 mask 掉。该副本不写回 shared ledger，只用于 action feasibility；真正 commit/rollback 仍由 MultiServiceDynamicEnv 独占。

本地 20-episode 诊断（`seed=100`、`risk=20`、batch=5、critic lr=`0.003`、critic updates=`20`）中，Upper 的 batch pre-update EV 为 `-0.004、+0.190、+0.189、+0.216`，critic 已不再是零价值 baseline。总计 404 次执行事件、156 次迁移提交，其中 309 次是 delayed pending 执行。失败最集中的模式是 delayed `link-only` 的 `fixed_node_infeasible`（79 次）：延迟期间原宿主已失效，而 link-only 不能修复节点。

因此新增 stale pending guard：link-only pending 在 due slot 使用最新 residual graph 检查固定节点；partial/full pending 则先重新 localized，再检查是否仍存在第一个可行 Lower action。任何 scope 若已无法启动，均记录 `pending_cancelled` 与 `pending_stale_cancelled`，不调用必然失败的 executor。Trainer 对 `pending_cancelled` 写入 `failure=-1`，并通过 `origin_decision_id` 归因给创建该 pending 的 Upper action。下一时隙服务会按最新预测重新进入 Selector/HAC；路由可行性仍由后续确定性 routing 验证。

## 常用命令

```bash
python -B train_multiservice_hac.py \
  --stgcn-checkpoint artifacts/stgcn_h3/stgcn_best.pt \
  --episodes 50 \
  --steps 100 \
  --seed 100 \
  --reward-risk-weight 20 \
  --update-batch-episodes 5 \
  --calibration-eval-seeds \
  --validation-seeds 1200 1201 1202 1203 1204 1205 1206 1207 1208 1209 \
  --validation-interval 10 \
  --output artifacts/hac_risk20_batch5_stability
```

```bash
python -B smoke_hac_worker.py
```
