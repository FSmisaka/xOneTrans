# Continuous Spectral OneTrans

## 1. 实验目标

### 1.1 研究问题

实验 3 已证明：通过 NER 正则可以把两条 OneTrans 分支的最终序列表示推向高 NER 和低 NER，但极端 NER 分化并未带来稳定的融合收益。本实验不再训练多套独立 OneTrans，也不再把最大化/最小化 NER 作为训练 Loss，而是研究：

> 能否从同一条 OneTrans 序列表示出发，构造一族具有连续不同 NER 的谱变换表示，并让模型学习如何融合这些不同谱形态对应的预测结果？

核心思想：

1. OneTrans 的前部 Backbone 只计算一次，得到进入最后一层 OneTrans Block 前的序列表示。
2. 在固定的训练集参考谱基上，对该表示施加连续参数化的谱变换，得到不同压缩程度的虚拟链路。
3. 多个谱变换共享 OneTrans 最后一层和原 CTR Tower 的参数，不复制完整 OneTrans。
4. 使用固定积分权重或可学习的权重函数融合不同谱链路的预测；另实现表示级融合，比较两种融合位置。
5. 全程沿用 CTR 监督，不加入 NER 正则。NER 用于控制/诊断谱变换，不作为直接优化目标。

本实验是对“共享 OneTrans + 连续谱变换族”的受控研究，不等价于多条参数完全独立、从头训练的 OneTrans。该设计有意共享参数，以隔离“谱形态变化”本身的作用，并大幅降低参数量。

### 1.2 主要假设

- H1：改变谱参数 `alpha` 能够连续控制参考序列表示的 NER。
- H2：不同 `alpha` 的 CTR 预测能力不必单调变化；可能存在中间最优区间。
- H3：对不同 `alpha` 的预测进行可学习加权，可能优于固定单一 `alpha` 或均匀融合。
- H4：表示级融合与预测级融合不等价；需要通过实验判断哪一种更有效。

所有假设均为待验证问题，不预设高 NER 或低 NER 更优。

---

## 2. 数学定义

### 2.1 沿用项目现有 NER 定义

必须严格沿用 `src/experiments/diagnostics.py::normalized_effective_rank` 的定义，不得自行改成奇异值平方：

1. 收集有效 S-token，按行拼接为 `X ∈ R^{N×D}`。
2. 按列中心化：`Xc = X - mean(X, dim=0)`。
3. SVD：`Xc = U diag(sigma) V^T`。
4. `p_i = sigma_i / sum_j(sigma_j)`。
5. `ER = exp(-sum_i p_i log(p_i))`。
6. `NER = ER / D`，其中 `D = token_dim`，不是 `min(N,D)`。

训练与评估的 NER 口径必须保持一致。参考实现：

- `src/experiments/diagnostics.py::normalized_effective_rank`
- `src/models/ner_regularization.py::differentiable_ner_stats`

### 2.2 连续谱变换族

对每个 seed，先从已有 OneTrans checkpoint 提取**进入最后一个 OneTrans Block 之前**的有效 S-token。只使用训练集校准样本，拼接得到参考矩阵：

`X_ref ∈ R^{N_ref×D}`

计算训练集参考均值与 SVD：

`mu_ref = mean(X_ref, dim=0)`

`Xc_ref = X_ref - mu_ref`

`Xc_ref = U diag(sigma) V^T`

定义连续压缩参数 `alpha > 0`：

`p_i(alpha) = sigma_i^alpha / sum_j sigma_j^alpha`

` sigma_i(alpha) = (sum_j sigma_j) * p_i(alpha)`

因此：

- `alpha < 1`：谱分布更均匀，NER 倾向提高。
- `alpha = 1`：恢复原始谱分布。
- `alpha > 1`：谱分布更集中，NER 倾向降低。

对每个 `alpha` 定义：

`r_i(alpha) = sigma_i(alpha) / sigma_i`

`T_alpha = V diag(r_i(alpha)) V^T`

对任意输入 S-token `h`，变换为：

`h_alpha = mu_ref + (h - mu_ref) T_alpha`

Padding 位置在变换后重新置零。

在参考矩阵上有：

`Xc_ref T_alpha = U diag(sigma(alpha)) V^T`

因此参考矩阵的奇异值谱被精确变为 `sigma(alpha)`（数值截断误差除外），且 `alpha=1` 时 `T_alpha=I`。

**重要：** 该变换矩阵由训练集参考谱基构造，并在整个训练、验证、测试及推理过程中固定使用。不得在每个 mini-batch 上重新做 SVD，否则同一样本的表示会依赖同 batch 的其他样本，训练/推理结果也会依赖 batch composition。

### 2.3 数值稳定性

- 校准 SVD 使用 FP64 或稳定 FP32；保存 `mu_ref`、`V`、`sigma`、`T_alpha`。
- 仅对 `sigma_i > sigma_max * rank_relative_tolerance` 的谱方向计算比例；其余方向的比例置零。
- `rank_relative_tolerance` 默认 `1e-6`，配置化。
- 用 log-domain 计算 `sigma_i^alpha`：`softmax(alpha * log(sigma_i))`，避免幂运算溢出。
- `alpha=1` 必须走 identity 分支，直接返回原始 `h`，确保数值上严格无变换。
- 对每个 `alpha` 保持参考中心化表示的核范数 `sum(sigma)` 不变；不得额外做逐 alpha 的 LayerNorm 或任意幅值归一化。
- 对低秩/零谱方向、NaN、Inf 及空校准矩阵编写单元测试。不得通过把零谱方向加成非零值来人为抬高 NER。

### 2.4 连续融合与有限求积

令 `t = log(alpha)`，在 `t ∈ [log(alpha_min), log(alpha_max)]` 上定义条件密度：

`pi_phi(t | c) = exp(s_phi(c,t)) / integral exp(s_phi(c,u)) du`

其中 `c` 是不含 label 的模型输入上下文。默认在 log-alpha 空间积分，意味着对乘法尺度采用均匀基准测度。

工程上使用复合 Simpson 求积近似连续积分。默认 7 个等距 log-alpha 节点：

`alpha_nodes = [0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0]`

归一化求积权重：

`quad_weights = [1, 4, 2, 4, 2, 4, 1] / 18`

对于每个样本 `b`：

`w_bk = softmax(s_phi(c_b, log(alpha_k)) + log(quad_weight_k))`

其中 `sum_k w_bk = 1`。如果 `s_phi=0`，就退化为固定 Simpson 权重。

有限 K 节点是对连续谱积分的数值近似，不代表模型真的运行无穷条独立 OneTrans。

---

## 3. 为什么要在最后一个 OneTrans Block 前变换

当前 `OneTransDiagnostic.forward` 的预测 readout 为：

`readout = ns_tokens.flatten(start_dim=1)`

`logit = tower(readout)`

CTR Tower 直接读取最后一层的 NS tokens；最后一层 S-stream 输出本身不直接进入 CTR Tower。NS tokens 通过 OneTrans 最后一层 Attention 的 K/V 读取 S-stream。

因此，谱变换必须插入在：

**前 `num_layers - 1` 层完成之后、最后一个 OneTrans Block 执行之前。**

这样每个 `alpha` 链路的变换后 S-token 会被最后一层 NS Query 读取，影响 NS readout 和 CTR 预测。

禁止只在最后一层结束后修改 S-token、却不重新执行 NS 对 S 的 Attention；这种做法不会有效改变当前 CTR readout。

---

## 4. 模型结构

### 4.1 共享 Backbone 与虚拟谱链路

模型结构：

`Input -> OneTrans Blocks [0 ... L-2] -> S_pre, NS_pre`

然后对每个 `alpha_k`：

`S_pre -> SpectralTransform(alpha_k) -> Shared OneTrans Block [L-1] -> Shared CTR Tower -> logit_k`

所有 `alpha_k` 链路：

- 共享相同的前 `L-1` 个 OneTrans Blocks；
- 共享同一个最后 OneTrans Block；
- 共享同一个原始 CTR Tower；
- 不创建 K 份独立 OneTrans 参数；
- 每条链路仅在最后一个 Block 的输入 S-token 谱形态上不同。

每次调用最后一层时，必须从同一个 `NS_pre` 开始，不能把上一条 alpha 链路更新后的 NS tokens 传给下一条链路。每个分支都使用独立的张量结果，但共享参数对象。

### 4.2 模型初始化与冻结策略

为保持全局参考谱基有效，Exp4 第一阶段使用两阶段训练：

1. 加载 Exp3 的 `single_baseline` 最佳 checkpoint。
2. 用该 checkpoint 在训练集上计算 `S_pre` 校准矩阵及谱基。
3. 冻结 Embedding、Tokenizer 和前 `L-1` 个 OneTrans Blocks。
4. 只训练最后一个 OneTrans Block、原 CTR Tower，以及（若启用）alpha 权重网络。

每个 condition 从相同 seed 的同一个 baseline checkpoint 初始化，避免不同条件有不同的前缀表示。

基线对照也必须遵守相同冻结策略：`alpha=1`、不启用谱变换、只训练最后 Block 和 Tower。该条件命名为 `baseline_frozen_prefix`。

默认 `trainable_last_n_layers=1`。第一版禁止解冻谱变换点之前的 Backbone；否则参考谱基会随训练漂移，需另行设计重估/EMA 机制，不属于本次 Exp4。

### 4.3 共享最后一层执行接口

对 `src/models/onetrans_diagnostic.py::OneTransBlock` 做最小重构，增加可分段调用能力，例如：

- `forward_prefix(s_tokens, ns_tokens, mask, end_layer)`：执行前 `end_layer` 个 Block，返回 `s_pre, ns_pre, mask`。
- `forward_one_layer(s_tokens, ns_tokens, mask, layer_idx)`：使用指定共享 Block 执行一次更新。

必须保留原有 residual、LayerNorm、MixedMHA、MixedFFN、mask 和 causal attention 逻辑。

原来的完整 `forward` 路径在不传 spectral transform 时必须与重构前保持数值一致（FP32 容差 `1e-6`）。

---

## 5. 融合方式

实现以下四种融合模式。

### 5.1 `uniform_expert_mix`

- 使用固定 Simpson 权重 `quad_weights`。
- 每个 alpha 节点分别执行共享的最后 OneTrans Block 和 CTR Tower。
- 节点概率：`p_k = sigmoid(logit_k)`。
- 融合概率：`p_mix = sum_k quad_weight_k * p_k`。

### 5.2 `global_learned_expert_mix`

学习一个与样本无关的连续权重函数：

`s_phi(t)`

`w_k = softmax(s_phi(log(alpha_k)) + log(quad_weight_k))`

所有样本共享同一组 alpha 权重。

### 5.3 `contextual_learned_expert_mix`（主方法）

使用每个样本的上下文动态决定 alpha 权重。

上下文定义：

- `s_pool = MaskedMeanPool(S_pre)`，必须忽略 padding；
- `ns_flat = flatten(NS_pre)`；
- `c = concat(s_pool, ns_flat)`。

权重网络：

`s_bk = MLP(concat(c_b, log(alpha_k), log(alpha_k)^2))`

`w_bk = softmax_k(s_bk + log(quad_weight_k))`

禁止使用 label、未来行为或 test 统计量生成权重。

每个 alpha 节点的概率融合：

`p_mix_b = sum_k w_bk * sigmoid(logit_bk)`

这是概率级 Mixture-of-Experts。报告时以 `p_mix` 计算 AUC 和 LogLoss。

同时可额外报告 `sigmoid(sum_k w_bk * logit_bk)` 作为 weighted-logit ablation，但不能与概率混合结果混为一谈。

### 5.4 `contextual_representation_mix`

该模式用于比较“先融合表示”与“先独立预测再融合预测”是否等价。

先使用同一 contextual gate 计算权重 `w_bk`，再融合最后一层之前的 S 表示：

`S_mix_b = sum_k w_bk * S_alpha_k_b`

然后只执行一次共享最后 OneTrans Block 和 CTR Tower，得到 `logit_mix`。

由于最后 OneTrans Block 含 Attention 和 FFN 非线性，一般有：

`f(sum_k w_k S_alpha_k) != sum_k w_k f(S_alpha_k)`

所以该模式只是表示级连续积分近似，不得宣称与 expert probability mixture 数学等价。

---

## 6. 训练目标

### 6.1 主损失

对所有 mixture 条件，以概率融合输出为主要监督：

`L_mix = BCE(p_mix, label)`

实现时将 `p_mix` clamp 到 `[1e-7, 1 - 1e-7]` 后计算 BCE，或使用数值稳定的等价 log-probability 计算。

对 `fixed_alpha` 单链路条件：

`L = BCEWithLogits(logit_alpha, label)`

### 6.2 节点辅助 CTR Loss

为了让每个 alpha 节点在融合训练中都保持可评价的独立预测能力，提供可配置的节点辅助损失：

`L_nodes = sum_k quad_weight_k * BCEWithLogits(logit_k, label)`

`L_total = L_mix + node_aux_loss_weight * L_nodes`

默认 `node_aux_loss_weight=0.1`，并提供 `0.0` 消融配置。该辅助项只用于 mixture-expert 模式；representation-mix 模式不使用该项。

不加入 NER Loss，不加入额外的高/低 NER 约束，不因观察到 AUC 变化而动态修改 alpha。

### 6.3 优化和早停

- 沿用项目 `config/onetrans.py` 的 dense optimizer、learning rate、batch size、early stopping 等训练约定。
- 冻结参数不得加入 optimizer。
- 训练、validation、test 数据切分沿用当前 KuaiRand 预处理，不重新划分。
- mixture 条件以 `valid_mix_auc` 作为 early-stopping 主指标；fixed-alpha 条件以对应 `valid_auc` 作为主指标。
- Test 只在 best validation checkpoint 确定后评估一次。
- 默认 seeds `[2026, 2027, 2028]`。

---

## 7. 实验条件

### 7.1 固定 alpha 单链路谱扫描

每个 alpha 独立启动一次训练；同 seed 均从同一个 Exp3 `single_baseline` checkpoint 初始化，冻结相同前缀，只训练最后 Block 与 CTR Tower。

`FIXED_ALPHA = [0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0]`

其中 `alpha=1.0` 即 `baseline_frozen_prefix`，必须与 identity transform 完全一致。

该扫描用于建立 alpha / NER / AUC / LogLoss 的关系，不预设单调性。

### 7.2 连续融合条件

| Condition | 融合方式 | 权重 | 最后一层调用次数 |
|---|---|---|---:|
| `uniform_expert_mix` | 预测级概率融合 | 固定 Simpson 权重 | K |
| `global_learned_expert_mix` | 预测级概率融合 | 全局可学习连续权重函数 | K |
| `contextual_learned_expert_mix` | 预测级概率融合 | 样本条件可学习连续权重函数 | K |
| `contextual_representation_mix` | 表示级融合 | 样本条件可学习连续权重函数 | 1 |

此外运行 `baseline_frozen_prefix`。

`K=7` 为默认值。若显存或延迟过高，可以配置 K=5，但必须同步生成对应的 log-alpha 节点和归一化求积权重，并在结果中记录。

### 7.3 预算控制

- 先运行 KuaiRand-1K 完整实验。
- `--debug` 只用于形状、梯度、数据流和速度测试，不作为结论。
- 1K 结果通过后，才考虑 27K 验证。
- 所有条件共享相同 seed 的 baseline checkpoint 和校准谱基。

---

## 8. 评估指标与分析

### 8.1 CTR 指标

对 `baseline_frozen_prefix`、每个固定 alpha、每个 mixture 条件报告：

- Validation AUC / LogLoss。
- Test AUC / LogLoss。
- 相对 `baseline_frozen_prefix` 的 paired delta。

对 expert mixture 额外报告每个 alpha 节点的独立 AUC / LogLoss，以及融合后的 AUC / LogLoss。

### 8.2 NER 指标

必须区分以下指标：

1. **Reference analytic NER**：根据校准 SVD 的 `sigma(alpha)` 直接计算理论 NER。
2. **Calibration empirical NER**：将 `T_alpha` 应用到参考矩阵，再调用项目现有 `normalized_effective_rank` 计算，验证理论公式与实际实现一致。
3. **Validation/Test global NER**：对每个 alpha 变换后的有效 S-token，按项目原有方法跨样本拼接、全局中心化并计算 NER。
4. **Post-final-layer NER**：记录每个 alpha 经过共享最后 OneTrans Block 后的最终 S-stream NER。
5. **Representation-mix NER**：仅对 `contextual_representation_mix` 计算实际融合表示的 global NER。

对于 prediction-level expert mixture，不存在唯一的“融合后序列表示矩阵”，不得把分支 NER 加权平均冒充融合表示 NER。可以报告各节点 NER 与样本平均权重，但需分开命名。

### 8.3 权重分析

对 global/contextual gate 报告：

- 每个 alpha 节点的平均权重、标准差、P10/P50/P90。
- 样本级权重熵：`-sum_k w_bk log(w_bk)` 的均值和分位数。
- 期望压缩参数：`E[alpha | x] = sum_k w_bk * alpha_k`。
- 不同历史长度区间、Target Item 类别（若字段可用）的平均权重及样本量。
- 权重与预测效果的分组关系仅作描述，不表述为因果关系。

### 8.4 成本指标

记录：

- 总参数量、可训练参数量。
- 训练时间、峰值显存。
- Test 推理总时间、ms/sample、samples/sec。
- 与 `baseline_frozen_prefix` 的额外延迟比例。

---

## 9. 工程实现与文件组织

沿用仓库现有结构，不重写数据管道，不复制 UniRank。

### 9.1 新增文件

```text
config/
  exp_4_continuous_spectral.py

src/models/
  spectral_transform.py
  continuous_spectral_onetrans.py

src/experiments/
  continuous_spectral.py

scripts/
  41_run_exp_4_continuous_spectral.sh
  42_summarize_exp_4.sh
  43_test_exp_4.sh

tests/
  test_exp_4.py
```

如需要可在 `src/experiments/run_experiment.py` 增加 `--exp 4` 路由，并在 `README.md` 中补充运行命令；不得破坏现有 1.x、2.x、3 的入口和结果格式。

### 9.2 需要复用的现有模块

- `src/models/onetrans_diagnostic.py::OneTransDiagnostic`：Embedding、Tokenizer、OneTrans Backbone、CTR Tower。
- `src/experiments/diagnostics.py::normalized_effective_rank`、`collect_states`：NER 与状态抽取。
- `src/experiments/dual_branch.py`：Exp3 的 checkpoint 加载、seed、DataLoader、训练日志、validation early stopping、metrics 汇总和 paired-delta 组织方式。
- `src/utils/metrics.py`：AUC / LogLoss。
- `config/onetrans.py`、`config/dataset.py`、`config/paths.py`：沿用已有配置。

Coding Agent 开发前必须重新检查当前仓库代码，不要只依据本 Spec 猜测函数签名。主要参考文件：

- 仓库 README：`https://github.com/FSmisaka/xOneTrans/blob/main/README.md`
- Exp1 的 NER 口径与项目约定：`https://github.com/FSmisaka/xOneTrans/blob/main/SPEC1.md`
- Exp2 的实验约定：`https://github.com/FSmisaka/xOneTrans/blob/main/SPEC2.md`
- OneTrans 模型实现：`https://github.com/FSmisaka/xOneTrans/blob/main/src/models/onetrans_diagnostic.py`
- NER 诊断实现：`https://github.com/FSmisaka/xOneTrans/blob/main/src/experiments/diagnostics.py`
- Exp3 训练与 checkpoint 保存：`https://github.com/FSmisaka/xOneTrans/blob/main/src/experiments/dual_branch.py`
- Exp3 分支模型：`https://github.com/FSmisaka/xOneTrans/blob/main/src/models/dual_branch_onetrans.py`
- Exp3 配置：`https://github.com/FSmisaka/xOneTrans/blob/main/config/exp_3_dual_branch_ner.py`

### 9.3 具体代码要求

**A. `src/models/spectral_transform.py`**

实现：

- `fit_spectral_basis(reference_states, reference_mask, rank_relative_tolerance)`：训练集有效 token 拼接、中心化、计算并保存 `mu_ref`、`V`、`sigma`。
- `build_spectral_transform(sigma, V, alpha, ...)`：生成 `T_alpha`。
- `apply_spectral_transform(states, mask, mu_ref, T_alpha)`：对 batch 中有效 S-token 应用同一个固定变换，Padding 输出置零。
- `analytic_ner_from_spectrum(sigma, alpha, token_dim)`。

谱基拟合只允许使用 train split，不得读取 validation/test 标签或统计量。谱基和变换矩阵应作为 checkpoint/artifact 保存，并记录来源 seed、checkpoint、校准样本索引、token 数量和配置。

**B. `src/models/onetrans_diagnostic.py`**

以最小改动为原则，为 `OneTransBlock` 增加 prefix / single-layer 执行接口。保持原始完整 forward 在无变换时数值等价。

不得更改 OneTrans 的 Attention mask、token 组织、FFN、Tokenizer 或 CTR Tower 结构。

**C. `src/models/continuous_spectral_onetrans.py`**

实现共享前缀、谱变换族、共享最后 Block、共享 Tower、权重网络和两种融合模式。

Forward 至少返回：

- `logits_by_alpha: [B,K]`（expert 模式）。
- `weights: [B,K]`。
- `mix_prob: [B]`。
- `mix_logit: [B]`。
- `alpha_nodes`、`quad_weights`。
- 用于诊断的 `s_pre`、`s_alpha` 或可选状态；正常训练不保留不必要的中间状态。

representation-mix 模式返回 `mix_logit` 以及真实的 `s_mix`，不伪造各节点独立 logits。

**D. `src/experiments/continuous_spectral.py`**

负责：baseline checkpoint 加载、train-only 校准谱基生成/读取、freeze 策略、condition 构建、训练、validation early stopping、test evaluation、NER 诊断、权重分析、成本统计和多 seed 汇总。

所有模型输出和日志按 condition/seed 隔离保存。`skip_existing` 只允许在配置与 checkpoint 元信息匹配时跳过。

### 9.4 配置文件

`config/exp_4_continuous_spectral.py` 导出 `CONFIG: Dict[str, Any]`。至少包括：

```python
CONFIG = {
    "experiment": "exp_4_continuous_spectral",
    "seeds": [2026, 2027, 2028],
    "baseline_checkpoint": "results/runs/exp_3_dual_branch_ner/single_baseline/seed_{seed}/checkpoint_best_mix.pt",
    "dataset": "1k",
    "calibration": {
        "split": "train",
        "max_tokens": 50000,
        "seed": 2026,
        "rank_relative_tolerance": 1e-6,
        "svd_dtype": "float64",
    },
    "spectral": {
        "alpha_nodes": [0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0],
        "quadrature_weights": [1/18, 4/18, 2/18, 4/18, 2/18, 4/18, 1/18],
        "integration_variable": "log_alpha",
    },
    "model": {
        "trainable_last_n_layers": 1,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.0,
        "fusion_modes": [
            "uniform_expert_mix",
            "global_learned_expert_mix",
            "contextual_learned_expert_mix",
            "contextual_representation_mix",
        ],
    },
    "training": {
        "node_aux_loss_weight": 0.1,
        "node_aux_loss_ablation": [0.0, 0.1],
        "freeze_prefix": True,
    },
    "evaluation": {
        "compute_global_ner": True,
        "compute_node_metrics": True,
        "compute_weight_stats": True,
        "compute_cost": True,
    },
}
```

所有路径必须通过 `config/paths.py` 派生；若 baseline checkpoint 缺失，给出明确错误并提示先运行 Exp3 `single_baseline`，不得静默随机初始化。

### 9.5 命令行入口

建议：

```bash
bash scripts/41_run_exp_4_continuous_spectral.sh --dataset 1k
bash scripts/41_run_exp_4_continuous_spectral.sh --dataset 1k --seeds 2026
bash scripts/41_run_exp_4_continuous_spectral.sh --dataset 1k --conditions baseline_frozen_prefix fixed_alpha uniform_expert_mix
bash scripts/42_summarize_exp_4.sh
bash scripts/43_test_exp_4.sh
```

正式实验可将 `1k` 替换成 `27k`。运行方式与现有 `scripts/31_run_exp_3_dual_branch.sh` 一致，优先复用 `src.experiments.run_experiment` 的入口风格。

---

## 10. 结果目录与输出文件

每次运行保存：

```text
results/runs/exp_4_continuous_spectral/<condition>/seed_<seed>/
  config.yaml
  spectral_basis.pt
  metrics_by_epoch.csv
  loss_curves.csv
  metrics_valid.json
  metrics_test.json
  checkpoint_best.pt
  model_summary.json
  alpha_node_metrics.csv
  alpha_weight_stats.json
```

汇总保存：

```text
results/metrics/exp_4_continuous_spectral/
  summary.json
  summary.csv
  paired_deltas.csv
```

图表保存到 `results/figures/`，至少包括：

1. `alpha_vs_ner.png`：alpha 与 reference / validation / test NER。
2. `alpha_vs_auc.png`：固定 alpha 节点的 AUC。
3. `alpha_vs_logloss.png`：固定 alpha 节点的 LogLoss。
4. `mixture_comparison.png`：baseline、fixed alpha、uniform/global/contextual expert mix、representation mix。
5. `weight_distribution.png`：全局/样本条件 alpha 权重分布。
6. `ner_layer_comparison.png`：不同 alpha 在变换点前后及最后层的 NER。

所有图沿用项目现有 `src/experiments/figures.py` 风格。

---

## 11. 必须通过的测试

在 `tests/test_exp_4.py` 中至少实现以下测试，并由 `scripts/43_test_exp_4.sh` 执行。

1. **Identity test**：`alpha=1` 时 `apply_spectral_transform` 输出与输入有效 token 完全一致（容差 `1e-6`）；Padding 仍为零。
2. **Reference spectrum test**：把全部 `X_ref` 变换后重新调用现有 `normalized_effective_rank`，与 `analytic_ner_from_spectrum` 的 NER 误差小于 `1e-4`。
3. **Monotonicity test**：对非平坦、满秩 toy spectrum，alpha 增大时 analytic entropy / NER 单调不增；alpha 减小时不低于 alpha=1 的 NER。
4. **Nuclear norm test**：各 alpha 变换后的 reference singular-value sum 与原值相对误差小于 `1e-4`。
5. **Batch-independence test**：同一个样本单独推理和与其他样本拼 batch 推理时，spectral transform 输出一致。
6. **Padding invariance test**：改变 padding 位置的原始 token 值不会改变有效位置的变换输出；Padding 输出为零。
7. **Backbone equivalence test**：无变换/alpha=1 时，分段 prefix + final-layer forward 与原始完整 OneTrans forward 的 logits、NS tokens、S states 一致（容差 `1e-6`）。
8. **Shared-parameter test**：所有 alpha 节点必须引用同一最后 OneTrans Block 和 CTR Tower，不允许复制 K 份模型参数。
9. **Mixture test**：权重非负且每个样本权重和为 1；若所有 gate score 为 0，weights 等于固定 Simpson 权重；若只有一个节点权重为 1，mix probability 等于该节点 probability。
10. **Gradient test**：CTR loss 能对最后 OneTrans Block、CTR Tower 和 gate（learned 模式）产生有限梯度；冻结前缀参数不更新。
11. **No leakage test**：谱基文件元数据的校准 split 必须是 train，且校准流程不接收 validation/test labels。

---

## 12. 结果解释原则

- 不预设高 NER 更好，也不预设低 NER 更好。
- 固定 alpha 的 AUC / LogLoss 曲线用于观察预测能力与谱形态的关系，不把相关性写成因果关系。
- 若 contextual gate 优于 uniform/global gate，说明样本条件化的谱融合可能有价值；仍需通过权重分布和消融解释。
- 若 representation mix 与 expert probability mix 不同，这是预期可能发生的结果，因为最后 OneTrans Block 是非线性的。
- 若模型权重集中在 alpha≈1，说明当前数据/模型未表现出利用强谱变换的收益，不得人为扩大 alpha 区间以追求更漂亮的权重分布。
- 若 mixture AUC 不优于最佳单 alpha，不代表连续融合实现错误；应检查各节点预测互补性、权重熵、节点辅助损失及验证集泛化。
- 若 NER 发生预期变化但 AUC 不变或下降，应作为有效研究结果记录，不得通过改变 NER 定义或只报告有利节点掩盖结果。

---

## 13. 实现顺序

按以下顺序开发，逐步通过测试：

1. 检查现有 Exp3 `single_baseline` checkpoint 加载格式及 `OneTransDiagnostic` 状态抽取接口。
2. 实现 train-only reference SVD、`T_alpha` 构造和 identity/monotonicity/nuclear-norm tests。
3. 将 OneTrans Backbone 拆分为 prefix 与单层执行接口，证明原模型数值不变。
4. 实现单 alpha 固定谱变换模型，完成 `baseline_frozen_prefix` 和 fixed-alpha sweep。
5. 实现共享最后层的 K-node expert mixture、uniform weights 和 probability fusion。
6. 实现 global/contextual continuous gate。
7. 实现 contextual representation mixture。
8. 完成多 seed 训练、validation early stopping、test 评估、NER/权重/成本分析与汇总。
9. 在 KuaiRand-1K 完整跑通后，再进行 27K 规模验证。

先保证数学变换、信息通路、alpha=1 identity 和训练/推理一致性正确，再开展完整实验。不得为了追求性能结果改变 OneTrans 主体结构或 NER 定义。
