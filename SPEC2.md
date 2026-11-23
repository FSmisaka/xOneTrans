# SPEC.md — Task-aware Representation Gating

## 1. 项目目标

基于现有 OneTrans + KuaiRand-1K 实验框架，设计并验证 **Task-aware Residual Representation Gate**。

核心研究问题：

> Effective Rank 与推荐预测性能并非单调关系。模型能否根据推荐任务，自适应地保留有预测价值的补充表示，而不是无差别地提高序列表示的 Effective Rank？

本阶段仅实现实验 2.1、2.2、2.3。

## 2. 项目约定

- 数据目录：`/data`，沿用现有 KuaiRand-1K 数据及数据划分。
- 模型及通用代码：`/src`。
- 配置文件：`/config`，每个 `.py` 配置文件导出 `CONFIG: Dict[str, Any]`。
- 运行脚本：`/scripts`。
- 沿用现有 OneTrans、数据处理、训练、评估和表示诊断代码，不复制或整体引入 UniRank 仓库。
- 保持与实验 1.1–1.3 一致的数据划分、训练流程、指标定义和 Effective Rank 计算口径。
- 所有新增超参数放入配置文件，不得硬编码。
- 代码保持简洁，避免不必要的异常处理、分支和注释。
- 数据、checkpoint、日志和实验产物均不得提交 Git。

## 3. 模块设计：Task-aware Residual Representation Gate

### 3.1 输入与插入位置

在 OneTrans 序列编码器输出处接入模块。

设序列表示为：

\[
H\in\mathbb{R}^{B\times L\times D}
\]

其中 \(B\) 为 batch size，\(L\) 为历史序列长度，\(D\) 为隐藏维度。

模块同时使用 target item embedding：

\[
E_t\in\mathbb{R}^{B\times D}
\]

模块输出维度仍为 \(B\times L\times D\)，直接接入原有预测头。不改变预测头、CTR loss 或样本构造。

### 3.2 前向计算

**主路径：**

\[
Z_c=H
\]

**补充路径：**

\[
Z_r=W_2\operatorname{GELU}
\left(W_1\operatorname{LN}(H)\right)
\]

其中 MLP 对每个 token 独立应用。

**序列摘要：**

\[
h_p=\operatorname{MaskedMeanPool}(H)
\]

Pooling 必须忽略 padding token。

**任务感知 Gate：**

\[
G=\sigma\left(\operatorname{MLP}
([h_p;E_t])\right)
\]

\[
G\in(0,1)^{B\times D}
\]

**最终表示：**

\[
\boxed{
Z_{\mathrm{out}}=
H+G[:,None,:]\odot Z_r
}
\]

Gate 对每个样本、每个隐藏维度自适应，并在该样本的历史 token 维度上共享。

### 3.3 实现要求

- 补充路径不得改变序列长度。
- MaskedMeanPool 忽略 padding；空历史样本应安全处理。
- Gate 只能使用推理时可获得的历史表示和 target item embedding，不得使用 label 或未来信息。
- Padding 位置遵循原模型 mask 约定。
- 不增加 Effective Rank 正则项或额外预测损失，沿用原 CTR loss。
- 模块支持关闭，关闭时应与原始 OneTrans baseline 一致。
- Gate 初始化、MLP 隐藏维度、dropout 等均配置化。

## 4. 实验 2.1：Task-aware Residual Gate

### 目的

验证任务感知补充通道能否改善 CTR 预测，而不要求 Effective Rank 必须提高。

### 实验组

| 实验组 | 结构 |
|---|---|
| Baseline | 原始 OneTrans |
| Task-aware Gate | 第 3 节完整模块 |

### 控制变量

- 相同数据划分、特征、训练样本、优化器、训练轮数、early stopping 和预测头。
- 至少使用 3 个相同的随机种子。
- 除新增模块及其必要参数外，不改变其他模型设置。
- 报告参数量；必要时报告训练和推理耗时。

### 评价指标

- Test AUC
- Test LogLoss
- Effective Rank
- History Shuffling Sensitivity（见第 6 节）

### 判断标准

主要判断标准是 AUC 提升或 LogLoss 降低，并且不显著损害历史行为利用能力。

Effective Rank 不要求上升。若预测性能提升而 Effective Rank 持平甚至下降，仍符合本研究假设。

## 5. 实验 2.2：Gate 机制对照与消融

### 目的

区分性能收益究竟来自补充路径本身，还是来自 Gate 的自适应控制。

### 实验组

所有实验组使用相同的主路径和补充路径，仅改变补充路径权重。

| 实验组 | 输出表示 | 含义 |
|---|---|---|
| No Residual | \(H\) | 原始 Baseline |
| Ungated Residual | \(H+Z_r\) | 始终全量加入补充表示 |
| Fixed Gate | \(H+\alpha Z_r\) | 使用固定标量 Gate |
| Learnable Gate | \(H+G[:,None,:]\odot Z_r\) | 任务感知 Gate |

Fixed Gate 至少测试：

\[
\alpha\in\{0.25,0.5,0.75\}
\]

若计算预算有限，可先运行 \(\alpha=0.5\)。

固定 Gate 不训练 Gate 参数。

### 控制变量与报告

- 所有实验组使用相同数据、训练协议和 seeds。
- 报告 Test AUC、Test LogLoss、Effective Rank、History Shuffling Sensitivity。
- 报告各组参数量。
- 重点比较 Learnable Gate 与 Ungated / Fixed Gate，而不仅仅与 Baseline 比较。

### 关键判断

- 若 Learnable Gate 优于 Ungated 和 Fixed Gate，说明自适应控制可能带来额外收益。
- 若 Fixed Gate 已达到相近效果，则复杂 Gate 的必要性不足，应优先考虑简单残差缩放。
- 若所有残差组均不优于 Baseline，应检查补充路径设计和插入位置，不得仅通过提高 Gate 值追求更高 Effective Rank。

## 6. 实验 2.3：Gate、表示几何与历史依赖分析

### 目的

分析 Gate 是否选择性利用补充表示，以及这种选择是否与表示几何、预测性能和历史行为利用程度相关。

### 6.1 表示几何分析

沿用实验 1.1–1.3 的 Effective Rank 计算方法和层位置，记录：

- Gate 输入表示 \(H\) 的 Effective Rank。
- Gate 输出表示 \(Z_{\mathrm{out}}\) 的 Effective Rank。
- 如现有诊断代码支持，额外记录补充表示 \(Z_r\) 的 Effective Rank。

报告各 seed 结果及均值、标准差。

不得为了实验结果更理想而更换 Effective Rank 定义。

### 6.2 History Shuffling Sensitivity

在 Test 集上进行推理，不重新训练。

对每个样本随机打乱有效历史 item 的顺序，保持以下内容不变：

- 历史 item 集合
- Target item
- 非序列特征
- Label
- 有效历史长度

计算：

\[
\Delta_{\mathrm{order}}
=
AUC(H_{\mathrm{original}})
-
AUC(H_{\mathrm{permuted}})
\]

同时报告原始与扰动后的 LogLoss 及其差值。

该指标衡量模型对历史顺序信息的敏感程度，不应直接等同于模型对全部历史信息的依赖程度。

打乱时只置换有效历史位置，不得改变 padding 位置或有效历史长度。

### 6.3 Gate 分布分析

在 Test 集记录 Gate 值，统计：

- Mean、Std、Min、Max
- P10、P25、P50、P75、P90
- 每个隐藏维度的平均 Gate 值
- 按历史长度分桶后的 Gate 均值和预测指标
- 如数据字段支持，按 Target Item 类别分组后的 Gate 均值和预测指标

绘制：

- Gate 值分布图
- Gate 均值与历史长度的关系图
- Gate 均值与分组 AUC 的关系图

分组分析必须报告样本量，避免过度解释小样本分组。

### 6.4 Gate 干预分析

对已训练的 Learnable Gate 模型进行 test-time 干预，不重新训练。

比较三种设置：

| 设置 | Gate | 输出 |
|---|---|---|
| Learned | 模型预测的 Gate | \(H+G\odot Z_r\) |
| Gate-Zero | 强制 \(G=0\) | \(H\) |
| Gate-One | 强制 \(G=1\) | \(H+Z_r\) |

比较三种设置的：

- Test AUC
- Test LogLoss
- Effective Rank

用于判断模型学习到的 Gate 是否优于完全关闭或完全打开补充路径。

### 6.5 结果解释原则

- 不将高 Effective Rank 直接解释为信息更多或表示更好。
- 不将低 Effective Rank 直接解释为有害坍塌。
- 结合表示几何、预测性能、历史顺序敏感性和 Gate 干预结果讨论任务意义。
- Gate 与 AUC 的相关性属于观察性证据，不得表述为因果关系。
- 若 Learnable Gate 提升预测性能但没有提高 Effective Rank，可视为支持“任务相关的信息选择优于无差别提高秩”的证据，仍需结合干预实验判断。

## 7. 配置要求

新增配置文件建议：

- `config/exp_2_1_task_aware_gate.py`
- `config/exp_2_2_gate_ablation.py`
- `config/exp_2_3_gate_analysis.py`

每个文件导出 `CONFIG: Dict[str, Any]`。

至少包含以下配置类别：

```python
CONFIG = {
    "experiment": {...},
    "data": {...},
    "model": {
        "use_residual_gate": True,
        "residual_hidden_dim": ...,
        "gate_hidden_dim": ...,
        "gate_dropout": ...,
        "gate_init_bias": ...,
        "fixed_gate_alpha": ...,
    },
    "training": {
        "seeds": [...],
    },
    "evaluation": {
        "compute_effective_rank": True,
        "compute_history_shuffle": True,
        "compute_gate_statistics": True,
    },
}
```

以上为配置结构示意。实际字段名应与项目现有配置系统一致，不另建不兼容的配置接口。

所有新增超参数、seeds 和分析开关均须配置化。

## 8. 脚本与输出

建议新增：

- `scripts/run_exp_2_1.py`
- `scripts/run_exp_2_2.py`
- `scripts/run_exp_2_3.py`

若项目已有统一训练入口，应扩展现有入口，不重复实现训练逻辑。

每个实验至少输出：

1. 每个 seed 的训练日志和最佳 checkpoint。
2. 每个 seed 的验证集、测试集指标。
3. 汇总 JSON：实验组均值、标准差及配置。
4. CSV：逐 seed 指标、Gate 统计、分组分析结果。
5. 图表：实验组指标对比、Effective Rank 对比、Gate 分布、Gate 分组分析。

输出文件包含实验编号、实验组名称和 seed，避免覆盖已有实验 1.x 结果。

原始数据、处理数据、checkpoint、日志和实验结果不得提交 Git。

## 9. 验收标准

### 代码验收

- Baseline 配置下关闭模块后，能够复现现有 OneTrans 行为。
- 新模块输入输出 shape 正确。
- Padding mask 处理正确。
- 不存在 Label 或未来信息泄漏。
- No Residual、Ungated、Fixed Gate、Learnable Gate 均可通过配置切换。
- 训练、评估和分析脚本可独立运行。
- 所有新增参数均由配置管理。

### 实验验收

- 2.1：完成 Baseline 与 Task-aware Gate 多 seed 对比。
- 2.2：完成 No Residual、Ungated Residual、Fixed Gate、Learnable Gate 对比。
- 2.3：完成 Effective Rank、History Shuffling Sensitivity、Gate 分布和 Gate 干预分析。
- 保存逐 seed 数据和汇总数据，不只保留均值。
- 报告明确区分实验观察、机制假设和因果结论。

## 10. 本阶段不做的事情

- 不最大化或最小化 Effective Rank，不增加 Effective Rank 正则项。
- 不引入 Reward Model、强化学习或辅助预测任务。
- 不实现 User / Context / TargetItem 独立 Gate；本阶段只研究序列表示上的 Residual Gate。
- 不更换 OneTrans 主干、数据集、数据划分或 CTR 预测头。
- 不复制 UniRank 仓库，不重写已有数据处理和训练框架。