# Dual-Branch OneTrans with High-/Low-NER Regularization

## 1. Objective

在 OneTrans 上构建两条**参数独立、输入相同**的分支：

- **High-NER Branch**：通过辅助损失最大化序列流表示的 NER。
- **Low-NER Branch**：通过辅助损失最小化序列流表示的 NER。
- **Mixed Head**：拼接两条分支的 CTR readout，经 MLP 输出 CTR 预测。

三个 CTR 输出均参与训练，并分别在验证集、测试集报告 AUC 和 LogLoss。

本实验的首要目标是研究 NER 与预测能力的关系，不预设高 NER、低 NER 或融合模型中的任何一个效果更好。

## 2. Non-goals

- 不修改 OneTrans 的注意力、token、Pyramid Stack、特征输入或其他主干结构。
- 不共享两条分支的 OneTrans 参数。
- 不以提升最终 AUC 为由改变 NER 的定义或计算对象。
- 不把 NER 作为唯一优化目标；三个 CTR Head 始终保留 CTR 监督。
- 第一阶段不加入额外 Gate、Cross-Attention、残差通道或特征路由模块。

## 3. Model Architecture

### 3.1 Two independent OneTrans branches

建立两个 OneTrans 实例：

```text
high_branch = OneTrans(...)
low_branch  = OneTrans(...)
```

要求：

1. 两个分支使用完全相同的 OneTrans 配置、输入字段、特征预处理、序列历史、attention mask 和 CTR readout 定义。
2. 参数完全独立，不共享 Embedding、Backbone、Norm 或其他参数。
3. 每个随机种子下，先初始化一个 OneTrans，再将其初始权重复制给另一个分支。两分支从相同初始化出发，之后独立更新。
4. 两个分支使用相同 batch、标签和数据顺序；Dropout 等随机算子正常独立采样。
5. 不改变原始 OneTrans 的 forward 语义。尽量复用现有模型实现，不复制、改写两套主干代码。

### 3.2 Three CTR heads

从每条分支提取与原始 OneTrans 完全相同的 CTR readout：

- `z_high`: High-NER Branch 的 CTR readout。
- `z_low`: Low-NER Branch 的 CTR readout。

分别建立独立 CTR Head：

```text
logit_high = ctr_head_high(z_high)
logit_low  = ctr_head_low(z_low)
```

融合分支：

```text
z_mix = concat(z_high, z_low, dim=-1)
logit_mix = ctr_head_mix(z_mix)
```

可选诊断（不参与训练）：计算 `logit_avg = 0.5 * (logit_high + logit_low)`，评估简单 logit averaging。它可以帮助判断 Mixed MLP 的收益是否超出简单分数融合。

`ctr_head_high` 和 `ctr_head_low` 应复用原始 OneTrans CTR Head 的结构。`ctr_head_mix` 使用轻量 MLP，默认结构：

```text
Linear(2 * readout_dim, readout_dim)
Activation(沿用项目默认激活函数)
Dropout(沿用项目默认配置)
Linear(readout_dim, 1)
```

若原始 CTR Head 的输入/输出接口不同，适配接口，但不得改变两个单分支 Head 的预测路径。三个 Head 输出 logits，统一使用原始 CTR 任务的 sigmoid/BCE 约定。

**梯度约定：** 默认允许 `logit_mix` 的 CTR Loss 反向传播到两个 OneTrans 分支；不对 `z_high`、`z_low` 做 detach。这样 Mixed Head 与两个分支端到端联合训练。单分支 CTR Loss 同时确保每条链路自身具备 CTR 预测能力。

## 4. NER Measurement and Regularization

### 4.1 NER calculation must match existing experiments

NER 的对象是 OneTrans 各层的 **S-stream hidden representations**，不是 NS-token、CTR readout、Attention Matrix 或混合后的向量。

必须复用项目现有实验中的 NER 定义和预处理约定，包括：

- 使用哪一层的 S-stream hidden state；
- 如何处理 batch、序列位置和 padding mask；
- 是否中心化；
- 奇异值或奇异值平方如何归一化；
- Effective Rank 如何归一化为 NER。

禁止另行定义一种数值不同的 NER 后直接与历史实验比较。

如果现有 NER 实现不可微，应实现与它数值等价的 PyTorch differentiable 版本，并先做一致性测试：对同一批表示，训练版 NER 与现有评估版 NER 的绝对误差应小于 `1e-4`（如数值精度不允许，需记录原因和实际误差）。

建议使用 `float32` 计算谱量。若用 Gram Matrix 加速，可对 `X.T @ X` 做对称化后用 `torch.linalg.eigvalsh`，并按照现有 NER 定义还原奇异值或其平方；需验证与现有 SVD 版本一致。

### 4.2 Which layers receive NER regularization

**默认只对最后一层 S-stream hidden representation（L）施加 NER loss。**

原因：最终层是当前研究中重点观察的 L4，先只控制最终层可以避免对 OneTrans 全部中间表示施加过强约束。

但评估时必须记录两条分支从输入到所有 Transformer 层的逐层 NER 曲线，包括 `backbone_input`。其中 `backbone_input` 不应受到 NER 正则化影响，应作为输入/实现一致性检查。

预留配置：`ner_target_layers: [last]`。后续可扩展为所有层，但第一阶段不得默认对所有层施加正则。

### 4.3 Differentiable spectral objective

令某分支目标层的 S-stream 表示矩阵为 `X`，按现有 NER 实现将有效 S-token 位置展平/组织成矩阵。对其谱量得到归一化概率分布 `p_i`，定义谱熵：

\[
H(X)=-\sum_i p_i\log(p_i+\epsilon).
\]

定义归一化谱熵：

\[
H_{norm}(X)=\frac{H(X)}{\log(r)},\qquad r=\min(N,D),
\]

其中 `N`、`D` 为现有 NER 矩阵的行数和列数；若现有实现对有效秩范围另有规定，必须沿用该规定。`H_norm` 仅用于稳定正则化尺度；它与当前 NER 的高低方向一致。评估结果仍报告现有定义的 NER。

两条分支的正则项：

\[
L_{high\_ner}=-H_{norm}(X_{high}),
\]

\[
L_{low\_ner}=+H_{norm}(X_{low}).
\]

因此，最小化总 Loss 时，High 分支倾向于增大 NER，Low 分支倾向于减小 NER。

实现要求：

- 仅对有效 S-token 位置计算，padding 不得进入谱统计。
- 训练期 NER 使用当前 mini-batch 计算；验证/测试期使用与历史实验相同的确定性评估流程。
- 对概率分布加入数值稳定项 `epsilon`，避免 `log(0)`。
- 不要对隐藏表示做额外归一化、白化或投影，除非历史 NER 计算本身如此。
- 记录每个 batch 的 NER 正则值和训练中实际观测到的 NER，确认正则方向正确。

## 5. Training Objective

三个 CTR Loss 均使用原始任务的标签、mask、样本权重和 BCE/LogLoss 实现：

\[
L_{ctr}=\frac{1}{3}\left(
L_{BCE}^{high}+L_{BCE}^{low}+L_{BCE}^{mix}
\right).
\]

总 Loss：

\[
\boxed{
L=L_{ctr}+\lambda_{high}L_{high\_ner}
+\lambda_{low}L_{low\_ner}
}
\]

其中 `lambda_high >= 0`，`lambda_low >= 0`。

- `lambda_high = 0` 时不施加高 NER 约束。
- `lambda_low = 0` 时不施加低 NER 约束。
- 两者均为 0 时，得到双分支、无 NER 正则的结构对照组。

对 NER 正则权重使用 warm-up：前 5%～10% 的训练 steps 从 0 线性增加至目标值，避免训练初期谱正则过强。

### 5.1 Lambda sweep

第一轮采用对称权重：

```yaml
lambda_high: [0, 1e-3, 1e-2, 1e-1]
lambda_low:  [0, 1e-3, 1e-2, 1e-1]
```

优先跑以下配置，不要求一次跑完笛卡尔积：

| Run | `lambda_high` | `lambda_low` | Purpose |
|---|---:|---:|---|
| dual_control | 0 | 0 | 双分支结构对照 |
| high_only | λ | 0 | 单独观察高 NER 约束 |
| low_only | 0 | λ | 单独观察低 NER 约束 |
| dual_split | λ | λ | 高/低 NER 双分支主实验 |

先用 `1e-2` 做 smoke test，再根据 NER 是否发生足够分离选择权重；若变化过弱或 CTR 明显不稳定，再扩大/缩小权重范围。记录 CTR Loss 与 NER Loss 的数值、梯度范数，避免仅凭 loss 数值判断权重是否合理。

## 6. Experimental Controls

至少比较以下条件：

1. **single_baseline**：原始单分支 OneTrans，使用现有训练脚本和配置。
2. **dual_control**：双 OneTrans + 三个 CTR Head，`lambda_high=lambda_low=0`。
3. **dual_split_ner**：双 OneTrans + 三个 CTR Head，一条最大化 NER，另一条最小化 NER。

随后按预算增加 `high_only`、`low_only`，用于判断哪一种 NER 约束导致变化。

所有条件必须保持相同：

- 训练/验证/测试切分；
- 样本、标签、负采样和样本权重；
- 特征预处理、历史长度和 mask；
- batch size、优化器、学习率策略、训练轮数/早停策略；
- 评估代码。

每个 seed 下，High/Low 分支从相同 OneTrans 初始权重复制而来。至少运行 3 个 seed；筛选出有意义的配置后，建议补充至 5 个 seed。

## 7. Evaluation and Reporting

### 7.1 CTR metrics

验证集和测试集分别报告以下三条预测链路：

| Prediction path | Required metrics |
|---|---|
| High-NER branch | AUC, LogLoss |
| Low-NER branch | AUC, LogLoss |
| Mixed head | AUC, LogLoss |

同时报告 `single_baseline` 的 AUC、LogLoss，便于与原始 OneTrans 比较。若启用 `logit_avg` 诊断，也报告其 AUC、LogLoss，但不将其计入三个主要预测链路。

### 7.2 NER metrics

对 High/Low 两个分支分别报告：

- `backbone_input` NER；
- 每个 Transformer layer 的 S-stream NER；
- 最后一层 NER；
- 各层 NER 相对 `single_baseline` 和 `dual_control` 的变化。

预期现象仅作为检查，不作为成功标准：High 分支 NER 应倾向上升，Low 分支 NER 应倾向下降。如果正则化未能实现预期谱变化，应报告实际结果，不得只报告 loss。

### 7.3 Checkpoint selection

- 使用验证集 `AUC_mix`（或项目既有的主验证指标）选择主 checkpoint；不得根据测试集选模型或调节 λ。
- 在同一个主 checkpoint 上，同时报告 High、Low、Mixed 三条预测链路的指标，确保三者可直接比较。
- 可额外报告每条 Head 自身最佳验证 checkpoint 的结果，但必须与同 checkpoint 结果分开标注，不可混用。

### 7.4 Statistical reporting

- 按 seed 报告均值和标准差，不只报告单次最好结果。
- 对核心模型比较报告配对 seed 差值。
- AUC 的小幅变化若与 seed 波动相当，应标记为不确定，不得宣称有效涨点。
- 记录训练时间、GPU 显存、参数量和推理 FLOPs/延迟；双分支模型计算成本预计高于单分支模型。

## 8. Logging and Artifacts

每次运行至少保存：

```text
run_dir/
  config.yaml
  metrics_by_epoch.csv       # train/valid: 三个 CTR heads + NER by layer
  metrics_test.json          # 三条 CTR 路径、逐层 NER、seed
  loss_curves.csv            # CTR loss、NER loss、total loss
  checkpoint_best_mix.pt
  model_summary.json         # 参数量、分支配置、训练耗时
```

`metrics_by_epoch.csv` 至少包括：

- `epoch`, `step`, `seed`；
- `loss_ctr_high`, `loss_ctr_low`, `loss_ctr_mix`；
- `loss_ner_high`, `loss_ner_low`, `loss_total`；
- `auc_high`, `logloss_high`；
- `auc_low`, `logloss_low`；
- `auc_mix`, `logloss_mix`；
- `ner_high_input`, `ner_high_l1 ... ner_high_lN`；
- `ner_low_input`, `ner_low_l1 ... ner_low_lN`。

## 9. Implementation Requirements and Tests

1. 保持原始单分支 OneTrans 路径可运行，不能因新实验破坏 baseline。
2. 检查两个分支参数无共享：除初始化时复制权重外，参数对象必须独立。
3. 检查三个 logits shape 均为 `[batch_size]` 或项目统一的 `[batch_size, 1]`。
4. 检查 `lambda_high=lambda_low=0` 时，NER 正则项不影响梯度；双分支训练仍正常。
5. 检查训练版可微 NER 与历史评估版 NER 的数值一致性。
6. 检查 High/Low 正则项梯度方向：单独优化 `-H_norm` 应增加 NER，单独优化 `+H_norm` 应降低 NER（在固定小批次上的单元测试）。
7. 检查 padding mask、生序列长度和空历史样本处理与原始 OneTrans 一致。
8. 检查 Mixed Loss 能向两个分支反向传播；High/Low NER Loss 只作用于各自分支。
9. 确保训练、验证、测试阶段均输出三条 CTR 路径的结果。

## 10. Expected Deliverable

提交可配置、可复现实验代码，支持：

- `single_baseline`、`dual_control`、`high_only`、`low_only`、`dual_split_ner`；
- High/Low NER loss 权重配置；
- 三条 CTR head 的联合训练与独立评估；
- 逐层 NER 评估；
- 多 seed 实验和统一结果汇总。

**核心交付不是某个预设的涨点结果，而是一张可靠的对照表：高 NER 分支、低 NER 分支、融合分支分别取得怎样的 AUC/LogLoss，以及它们实际形成了怎样的逐层 NER 结构。**
