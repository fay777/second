# 第二篇：预测感知弹性重配置的 MDP 形式化

<!-- 本文使用标准 Markdown 数学分隔符：行内公式为 $...$，独立公式为 $$...$$。 -->

## 1. 论文定位

题目建议采用：

**Proactive Elastic Virtual Network Reconfiguration via Topology Prediction for Dynamic Space-Air-Ground Integrated Networks**

这篇论文不再研究新到达虚拟网络请求的首次嵌入，而是研究**已运行虚拟网络在动态空天地一体化网络中的前瞻式重配置**。核心逻辑为：

运行中的虚拟网络  
-> 历史拓扑序列  
-> ST-GCN 预测未来风险图  
-> AC 迁移规划器  
-> 风险感知重配置执行器  
-> 新的稳定部署

这里的关键词应统一使用 **Reconfiguration**，而不是 **Re-embedding**。因为研究对象已经从一次性映射行为转为运行期动态管理。

## 2. 与第一篇的关系

第一篇和第二篇共用 Actor-Critic 训练范式，但问题定义不同。

- 第一篇：面向新到请求的首次嵌入，核心是 admission control + resource allocation。
- 第二篇：面向运行中业务的预测感知重配置，核心是 migration planning + reconfiguration execution。

第二篇不是简单复用第一篇算法，而是在同一 AC 框架下重新定义两个 MDP：

- 规划层 MDP：决定是否迁移、何时迁移、迁移范围。
- 执行层 MDP：在规划结果约束下生成新的节点/链路映射。

因此，第二篇的状态、动作、奖励、优化目标都与第一篇不同。

## 3. 系统模型

### 3.1 时变物理网络

将空天地一体化网络建模为时变图：

$$
\mathcal{G}^t = (\mathcal{N}^t, \mathcal{L}^t, \mathbf{X}_n^t, \mathbf{X}_l^t)
$$

其中：

- $\mathcal{N}^t$：时刻 $t$ 的物理节点集合，包括卫星节点、空中节点、地面节点。
- $\mathcal{L}^t$：时刻 $t$ 的物理链路集合，包括星间、星地、空地、地面链路。
- $\mathbf{X}_n^t$：节点特征矩阵。
- $\mathbf{X}_l^t$：链路特征矩阵。

节点特征可定义为：

$$
\mathbf{x}_{n_i}^t = [c_i^t, s_i^t, e_i^t, q_i^t, d_i]
$$

其中：

- $c_i^t$：剩余计算资源
- $s_i^t$：剩余存储资源
- $e_i^t$：能量或载荷余量
- $q_i^t$：排队状态或负载水平
- $d_i$：节点域类型，表示天基、空基、地基

链路特征可定义为：

$$
\mathbf{x}_{l_{ij}}^t = [b_{ij}^t, \tau_{ij}^t, p_{ij}^t, \delta_{ij}^t]
$$

其中：

- $b_{ij}^t$：剩余带宽
- $\tau_{ij}^t$：传播时延
- $p_{ij}^t$：链路质量损伤指标，如丢包率或误码倾向
- $\delta_{ij}^t$：链路可见持续时间

### 3.2 运行中虚拟网络

运行中的第 $k$ 个虚拟网络业务记为：

$$
s_k^t = (\mathcal{V}_k, \mathcal{E}_k, \mathcal{M}_k^t, \Theta_k, T_k^{rem})
$$

其中：

- $\mathcal{V}_k$：虚拟节点集合
- $\mathcal{E}_k$：虚拟链路集合
- $\mathcal{M}_k^t$：时刻 $t$ 的当前映射
- $\Theta_k$：业务 QoS/SLA 约束
- $T_k^{rem}$：剩余生命周期

当前映射写为：

$$
\mathcal{M}_k^t = (\mathcal{M}_{k,n}^t, \mathcal{M}_{k,l}^t)
$$

其中：

- $\mathcal{M}_{k,n}^t$：虚拟节点到物理节点的映射
- $\mathcal{M}_{k,l}^t$：虚拟链路到物理路径的映射

## 4. 拓扑预测与风险图构造

### 4.1 历史拓扑序列

以长度为 $W$ 的历史拓扑序列作为预测器输入：

$$
\mathbb{G}_t^{hist} = \{\mathcal{G}^{t-W+1}, \dots, \mathcal{G}^{t}\}
$$

采用 ST-GCN 学习时空相关性，预测未来 $H$ 个时隙的风险图序列：

$$
\hat{\mathbb{R}}_t = \{\hat{\mathcal{R}}^{t+1}, \dots, \hat{\mathcal{R}}^{t+H}\}
$$

其中：

$$
\hat{\mathcal{R}}^\tau = (\hat{\mathbf{r}}_n^\tau, \hat{\mathbf{r}}_l^\tau), \quad \tau \in [t+1, t+H]
$$

### 4.2 节点和链路风险

节点风险定义为：

$$
\hat{r}_{n_i}^\tau = f_n(\mathbf{x}_{n_i}^{t-W+1:t})
$$

链路风险定义为：

$$
\hat{r}_{l_{ij}}^\tau = f_l(\mathbf{x}_{l_{ij}}^{t-W+1:t})
$$

其中 $f_n$ 和 $f_l$ 由 ST-GCN 学得，可解释为未来一段时间内的失效、拥塞或性能退化倾向。

### 4.3 服务级风险

对已部署业务 $s_k$，定义未来窗口内的聚合风险：

$$
\hat{R}_k^t =
\lambda_1 \sum_{n \in \mathcal{M}_{k,n}^t} \bar{r}_{n}^{t:t+H}
+ \lambda_2 \sum_{l \in \mathcal{M}_{k,l}^t} \bar{r}_{l}^{t:t+H}
$$

其中：

- $\bar{r}_{n}^{t:t+H}$：节点在预测窗口内的平均风险
- $\bar{r}_{l}^{t:t+H}$：链路在预测窗口内的平均风险
- $\lambda_1, \lambda_2$：权重系数

这个服务级风险是后续迁移规划层的核心输入。

## 5. 两层 MDP 的边界

第二篇必须明确拆成两个层次：

### 5.1 迁移规划层

负责回答：

- 要不要动
- 什么时候动
- 动哪里

输出的是**迁移意图与迁移约束**，而不是新的映射方案。

### 5.2 重配置执行层

负责回答：

- 在已知迁移决策后
- 具体把哪些虚拟节点和虚拟链路映射到哪里
- 如何降低迁移代价和服务中断

输出的是**新的节点/链路映射方案**。

这两个层次的边界可以写成：

$$
\Pi_k^t = (y_k^t, \delta_k^t, \sigma_k^t)
$$

表示规划层输出，

$$
\mathcal{M}_k^{t,new} = (\mathcal{M}_{k,n}^{t,new}, \mathcal{M}_{k,l}^{t,new})
$$

表示执行层输出。

前者是 plan，后者是 mapping。

## 6. MDP 1：风险引导迁移规划

### 6.1 状态空间

对运行中业务 $s_k$，规划层状态定义为：

$$
s_{k,t}^{P} =
\left[
\mathbf{z}_{phy}^t,
\mathbf{z}_{svc,k}^t,
\mathbf{z}_{risk,k}^{t:t+H},
\mathbf{z}_{cost,k}^t,
\mathbf{z}_{sla,k}^t
\right]
$$

其中：

- $\mathbf{z}_{phy}^t$：当前物理网络全局状态嵌入
- $\mathbf{z}_{svc,k}^t$：业务 $k$ 当前部署状态嵌入
- $\mathbf{z}_{risk,k}^{t:t+H}$：由预测器得到的未来风险摘要
- $\mathbf{z}_{cost,k}^t$：业务迁移代价摘要
- $\mathbf{z}_{sla,k}^t$：业务 SLA 余量摘要

可进一步展开为：

$$
\mathbf{z}_{svc,k}^t =
[C_k^t, B_k^t, D_k^t, U_k^t, T_k^{rem}, F_k^t]
$$

其中：

- $C_k^t$：占用计算资源
- $B_k^t$：占用带宽资源
- $D_k^t$：当前端到端时延
- $U_k^t$：当前负载水平
- $T_k^{rem}$：剩余生命周期
- $F_k^t$：对全网碎片度的影响

### 6.2 动作空间

规划层动作定义为：

$$
a_{k,t}^{P} = (y_k^t, \delta_k^t, \sigma_k^t)
$$

其中：

- $y_k^t \in \{0,1\}$：是否迁移
- $\delta_k^t \in \{0,1,\dots,\Delta_{max}\}$：迁移时机，0 表示立即迁移
- $\sigma_k^t \in \{\text{link-only}, \text{partial}, \text{full}\}$：迁移范围

动作含义：

- `link-only`：只调整路径，不迁移虚拟节点
- `partial`：只迁移高风险子结构
- `full`：整张虚拟网络重配置

### 6.3 状态转移

若 $y_k^t = 0$，业务保持原部署继续运行。

若 $y_k^t = 1$，则在 $t+\delta_k^t$ 时刻触发执行层 MDP，并将 $\sigma_k^t$ 作为重配置约束输入。

### 6.4 奖励函数

规划层奖励定义为：

$$
r_{k,t}^{P}
=
\alpha_1 G_{k,t}^{safe}
- \alpha_2 C_{k,t}^{mig}
- \alpha_3 L_{k,t}^{sla}
- \alpha_4 O_{k,t}^{late}
$$

其中：

- $G_{k,t}^{safe}$：提前干预所避免的未来风险损失
- $C_{k,t}^{mig}$：预计迁移代价
- $L_{k,t}^{sla}$：业务 SLA 违规损失
- $O_{k,t}^{late}$：该迁未迁或迁移过晚的惩罚

也可写成更紧凑的形式：

$$
r_{k,t}^{P}
=
-\alpha_1 \hat{R}_{k}^{post}
-\alpha_2 C_{k}^{mig}
-\alpha_3 \mathbb{I}(\text{SLA violation})
-\alpha_4 \mathbb{I}(\text{failure before migration})
$$

其中 $\hat{R}_{k}^{post}$ 表示规划动作执行后的预测风险暴露。

### 6.5 优化目标

$$
\max_{\pi_P}
\mathbb{E}
\left[
\sum_t \gamma^t r_{k,t}^{P}
\right]
$$

该层优化的是**风险暴露、迁移代价、服务连续性**三者之间的长期平衡。

## 7. MDP 2：预测感知重配置执行

### 7.1 状态空间

执行层整体状态定义为：

$$
s_{k,t}^{E}
=
\left[
\mathbf{G}^t,
\hat{\mathbf{G}}^{t:t+H},
\mathcal{M}_k^t,
\sigma_k^t,
\mathcal{C}_k^t
\right]
$$

其中：

- $\mathbf{G}^t$：当前物理网络状态
- $\hat{\mathbf{G}}^{t:t+H}$：未来拓扑和风险预测摘要
- $\mathcal{M}_k^t$：当前映射
- $\sigma_k^t$：规划层给出的迁移范围
- $\mathcal{C}_k^t$：候选重配置资源集合

若按虚拟节点逐步执行，则 step 级状态写为：

$$
s_{k,t,m}^{E}
=
\left[
\mathbf{G}_{m}^t,
\hat{\mathbf{G}}^{t:t+H},
v_m,
\mathcal{M}_{k,m}^{partial},
\sigma_k^t
\right]
$$

其中：

- $v_m$：当前待重配置的虚拟节点
- $\mathcal{M}_{k,m}^{partial}$：当前部分映射结果

### 7.2 动作空间

为了保持与 `hrl-acra-main` 的 `hrl_ra` 结构一致，执行层可采用离散节点选择动作：

$$
a_{k,t,m}^{E} = p_m, \quad p_m \in \mathcal{N}_{cand}^t
$$

表示为当前虚拟节点选择新的宿主物理节点。

在此基础上，链路映射可以：

- 由环境按最短风险路径自动补全
- 或扩展为联合动作

$$
a_{k,t,m}^{E} = (p_m, \rho_m)
$$

其中 $\rho_m$ 表示路径或路由候选编号。

正文中建议优先采用第一种写法，以便与第一篇的下层 RA 框架保持一致。

### 7.3 状态转移

每执行一步动作，就更新：

- 节点剩余资源
- 链路剩余资源
- 当前部分映射
- 迁移中断代价

当所有目标虚拟节点和虚拟链路都完成重配置后，输出：

$$
\mathcal{M}_k^{t,new}
=
(\mathcal{M}_{k,n}^{t,new}, \mathcal{M}_{k,l}^{t,new})
$$

### 7.4 奖励函数

执行层奖励定义为：

$$
r_{k,t,m}^{E}
=
\beta_1 Q_{k,t,m}^{emb}
- \beta_2 C_{k,t,m}^{inst}
- \beta_3 R_{k,t,m}^{future}
- \beta_4 I_{k,t,m}^{srv}
- \beta_5 F_{k,t,m}
$$

其中：

- $Q_{k,t,m}^{emb}$：当前重配置映射质量
- $C_{k,t,m}^{inst}$：即时迁移成本
- $R_{k,t,m}^{future}$：未来风险暴露
- $I_{k,t,m}^{srv}$：服务中断损失
- $F_{k,t,m}$：资源碎片化损失

可进一步解释为：

- 映射到低风险、高余量节点时给正奖励
- 引起长路径、高中断、大迁移量时给负奖励
- 若新映射在未来窗口内仍处于高风险区域，则继续惩罚

### 7.5 终止条件

执行层 episode 的终止条件包括：

- 目标重配置成功完成
- 无可行映射
- 超过允许迁移步数
- 超过时延预算或中断预算

### 7.6 优化目标

$$
\max_{\pi_E}
\mathbb{E}
\left[
\sum_m \gamma^m r_{k,t,m}^{E}
\right]
$$

该层优化的是**可行重配置质量、服务连续性和未来稳定性**。

## 8. 与 `hrl-acra-main` 的对应关系

第二篇的方法可以直接参考 `hrl-acra-main` 的分层结构，但必须重新定义语义。

### 8.1 对应关系

- 第一篇 `hrl_ac`
  - 输入：当前物理网络 + 新到达 VNR
  - 输出：接纳或拒绝
- 第二篇规划层 AC
  - 输入：当前物理网络 + 已运行业务 + 未来风险图
  - 输出：是否迁移、何时迁移、迁移范围

- 第一篇 `hrl_ra`
  - 输入：当前物理网络 + 当前待映射虚拟节点
  - 输出：宿主物理节点
- 第二篇执行层 AC
  - 输入：当前物理网络 + 当前映射 + 未来风险图 + 规划结果
  - 输出：新的宿主物理节点或路径选择

### 8.2 第二篇新增的关键状态量

相较于第一篇，第二篇应显式加入：

- future risk embedding
- migration cost
- service interruption budget
- remaining lifetime
- current mapping pattern

因此，虽然底层实现仍可参考 `hrl-acra-main` 的 PPO + GNN + sequence decision 方式，但问题本身已发生变化。

## 9. 论文中的三个贡献点写法

建议写成：

1. 提出一个基于拓扑预测的主动弹性虚拟网络重配置框架，用于动态空天地一体化网络中运行中业务的前瞻式调整。
2. 设计一个风险引导的迁移规划机制，联合决策是否迁移、迁移时机和迁移范围，以在未来风险暴露与迁移代价之间实现平衡。
3. 设计一个预测感知的弹性重配置执行机制，在当前和未来网络状态约束下生成新的低风险节点/链路映射方案，并降低服务中断和资源碎片化。

这三个点分别对应：

- 框架
- Planning 层
- Execution 层

边界清晰，不会被审稿人认为把同一件事拆成两个贡献点。

## 10. 方法章节建议结构

第二篇的算法章节建议按以下结构展开：

### 4.1 Framework Overview

- 运行中虚拟网络
- 历史拓扑序列
- ST-GCN 风险预测
- 迁移规划
- 重配置执行
- 稳定部署

### 4.2 Topology Prediction and Risk Graph Construction

- ST-GCN 输入输出
- 节点风险和链路风险构造
- 服务级风险聚合

### 4.3 Risk-guided Migration Planning

- 规划层状态空间
- 规划层动作空间
- 规划层奖励函数
- 规划层优化目标

### 4.4 Prediction-aware Elastic Reconfiguration Execution

- 执行层状态空间
- 执行层动作空间
- 执行层奖励函数
- 执行层输出的新映射

### 4.5 Training and Online Inference

- 先训练 ST-GCN 预测器
- 再训练规划层 AC
- 最后训练执行层 AC
- 在线部署时按串联方式调用

## 11. 可直接放进论文的总述段落

可直接写为：

> Different from conventional reactive re-embedding methods that only respond after failures or severe QoS degradation occur, the proposed framework proactively monitors the historical topology evolution of running virtual networks, predicts future risk graphs via ST-GCN, and performs elastic virtual network reconfiguration before service disruption emerges. To this end, we decompose the overall decision process into a migration planning MDP and a reconfiguration execution MDP, corresponding to "whether/when/where to reconfigure" and "how to reconfigure", respectively.

## 12. 下一步建议

下一步最值得继续细化的是两项：

1. 将第 6 节和第 7 节改写成正式论文小节语言，直接形成 `4.3` 和 `4.4`。
2. 补一版伪代码，将 ST-GCN、规划层 AC、执行层 AC 的调用顺序写清楚。

如果后续要继续落地到实现，可在当前目录再补三份文件：

- `第二篇_符号表.md`
- `第二篇_伪代码.md`
- `第二篇_实验设计.md`
