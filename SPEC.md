# Spec：OneTrans 序列表示退化诊断实验

## 项目概述与实验地图

本项目研究 OneTrans 统一排序模型中的序列表示是否会受到额外非序列特征的影响。项目使用 KuaiRand 用户历史行为序列、用户画像、上下文和目标物品属性，通过控制变量实验观察不同非序列输入条件下的序列表示变化。

本阶段只开展问题发现实验。Coding Agent 需要独立实现 OneTrans 和 KuaiRand 数据处理，并完成以下四组实验：

| 编号 | 实验 | 研究内容 | 主要控制变量 |
|---|---|---|---|
| 1.1 | 非序列特征数量 | 逐步增加输入模型的非序列字段数量，观察序列表示是否退化 | 固定特征质量、类型比例和历史长度 |
| 1.2 | 非序列特征质量 | 固定字段集合，通过字段值置换改变其有效信息量 | 固定字段数量、类型和历史长度 |
| 1.3 | 非序列特征类型 | 分别输入用户画像、上下文、目标物品属性及其组合 | 固定字段数量和历史长度，使用原始特征值 |
| 1.4 | 历史序列长度 | 改变输入模型的历史行为数量 | 固定非序列字段集合及其质量，可选执行 |

四组实验均使用相同的 OneTrans 骨干和训练协议。核心诊断指标为逐层 **Normalized Effective Rank**，并记录 Sequence Probe AUC、完整模型 CTR AUC 和 LogLoss。通过这些结果判断序列表示是否发生变化，以及变化是否影响序列信息的预测能力和整体推荐效果。

本阶段不实现门控模型、特征选择器或其他问题解决方法。后续门控实验 2.1—2.3 不属于本 Spec 的开发范围。

## 1. 任务范围

本阶段只实现问题发现实验 **1.1—1.4**。暂不实现任何门控模型及 2.1—2.3 实验。

研究问题：在 OneTrans 统一排序架构中，额外非序列特征的数量、信息质量和类型，以及历史序列长度，是否会影响模型内部的序列表示？

核心诊断指标为逐层 **Normalized Effective Rank**。同时记录序列表示的预测能力和完整模型 CTR 效果，避免仅凭表示维度变化得出结论。

## 2. 项目目录与文件约束

所有路径均相对项目根目录。

```text
.
├── README.md
├── SPEC.md
├── config/
│   ├── paths.py
│   ├── dataset.py
│   ├── onetrans.py
│   ├── sequence_probe.py
│   ├── exp_1_1_count.py
│   ├── exp_1_2_quality.py
│   ├── exp_1_3_type.py
│   └── exp_1_4_length.py
├── data/
│   ├── raw/
│   │   ├── KuaiRand-27K -> 原始数据软链接
│   │   └── KuaiRand-1K -> 原始数据软链接
│   ├── interim/
│   ├── processed/
│   ├── metadata/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   └── figures/
├── src/
│   ├── data/
│   ├── models/
│   ├── experiments/
│   └── utils/
└── scripts/
    ├── 00_link_raw_data.sh
    ├── 01_preprocess_kuairand.sh
    ├── 11_run_exp_1_1.sh
    ├── 12_run_exp_1_2.sh
    ├── 13_run_exp_1_3.sh
    ├── 14_run_exp_1_4.sh
    └── 20_summarize_results.sh
```

要求：

- 所有数据、缓存、模型权重、日志、指标和图表保存在 `data/`。
- 所有项目 Python 源码保存在 `src/`。项目中不克隆、不 vendoring UniRank，也不将 UniRank 作为运行时依赖。
- `config/` 中的 Python 文件只保存配置，不写模型或数据处理逻辑。
- 所有可执行脚本保存在 `scripts/`。脚本使用相对项目根目录的路径。
- `README.md` 只写环境安装、数据准备和实验运行命令，每条命令配一句简短说明。
- 代码保持凝练，避免冗余注释、异常捕获和重复条件判断。
- 不创建 Notebook。实验结果通过脚本生成 CSV、JSON 和 PNG。

### .gitignore

在项目根目录创建 `.gitignore`，避免将数据和运行产物提交到 Git。至少忽略以下内容：

```gitignore
/data/
.venv/
venv/
.env
__pycache__/
*.py[cod]
*.log
.DS_Store
.idea/
```

`data/` 下的原始数据软链接、预处理文件、缓存、模型权重、日志、指标、元数据和图表均保留在本地，不提交到 Git。源码、配置、脚本、README、Spec 和 `.gitignore` 应正常纳入版本控制。不要使用会忽略整个项目 `config/`、`src/` 或 `scripts/` 的规则。

每个配置文件统一使用以下格式：

```python
from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    ...
}
```

所有模型超参数、数据参数、实验变量、随机种子、采样规模和路径均从 `config/` 读取，禁止散落在源码或 shell 脚本中。

## 3. UniRank 与 OneTrans 实现

UniRank 仅作为只读参考实现。Coding Agent 需要理解原始实现，并在本项目中独立编写所需代码。项目中禁止克隆 UniRank 仓库、添加 Git Submodule、复制整段源文件或直接导入 UniRank 代码；运行实验不得依赖 UniRank 目录。

参考资料：

- 仓库：https://github.com/salmon1802/UniRank
- OneTrans 实现：`model_zoo/OneTrans.py`，https://github.com/salmon1802/UniRank/blob/main/model_zoo/OneTrans.py
- KuaiRand 预处理：`data/KuaiRand/preprocess_Kuairand_seq_action.py`，https://github.com/salmon1802/UniRank/blob/main/data/KuaiRand/preprocess_Kuairand_seq_action.py
- README 与配置说明：https://github.com/salmon1802/UniRank#readme

实现要求：

1. 在 `src/models/onetrans_diagnostic.py` 中独立实现项目所需的 OneTrans，包括特征 Embedding、Tokenizer、OneTrans Block、预测头及训练所需组件。实现应遵循原论文和 UniRank 参考实现的结构与计算逻辑。
2. 在 `src/data/kuairand_preprocess.py` 中独立实现 KuaiRand 预处理。参考 UniRank 的字段定义、时间切分和因果历史构造逻辑，适配本项目的数据路径及 1K / 27K 文件名。
3. 不直接复制 UniRank 的 Python 源码，不从 UniRank 导入模型、数据处理或训练模块。可以使用 PyTorch 等通用第三方依赖。
4. 对照参考实现核验关键模块的输入输出形状、Attention Mask、序列与非序列 Token 交互、Embedding 组织及预测头。不要自行简化或改变 OneTrans 的核心结构。
5. 在 `data/metadata/` 中记录参考仓库 URL、查阅时的上游 commit hash、对应文件路径和本项目实现版本，便于复现和核对。无需在项目中保存 UniRank 源码或 LICENSE 文件。

### 3.1 OneTrans 必须遵守的实现细节

1. 保留 OneTrans 的序列 Token 与非序列 Token 双流输入及其逐层交互结构。
2. `num_ns_token`、`tokenizer_type`、`token_dim` 等结构参数在所有实验中固定。
3. UniRank OneTrans 默认 `reduction_ratio=0.5`，会在不同层裁剪序列查询 Token。所有诊断实验必须设 `reduction_ratio=0.0`，保持各层序列位置一致。
4. OneTrans 的默认 `forward` 只返回预测结果。项目适配版增加 `return_sequence_states` 选项，保存 Backbone 输入及每层更新后的有效序列 Token 表示。正常训练时不保存中间状态；诊断时在 `eval()` 和 `no_grad()` 下开启。
5. 诊断状态必须同步返回有效序列 mask。Effective Rank 计算时剔除 Padding Token。
6. 目标物品 ID 是所有实验共有的必要输入，始终保留，不计入可变非序列特征数量。
7. 目标事件的行为标签和行为结果不得进入目标侧输入。历史 action 只包含目标时刻之前已发生的反馈。
8. OneTrans 会将目标物品表示拼入非序列 Tokenizer 输入。实验中允许控制目标物品的静态属性字段，但不得移除目标物品 ID。

## 4. 数据集与预处理

本阶段只使用 KuaiRand，不涉及 Taobao 或其他数据集。

原始数据根目录：

```text
KuaiRand-27K:
/Users/york/Documents/GitHub/hand-roll/data/raw_data/KuaiRand-27K

KuaiRand-1K:
/Users/york/Documents/GitHub/hand-roll/data/raw_data/KuaiRand-1K
```

原始目录中的 CSV 位于其 `data/` 子目录。

`data/raw/KuaiRand-27K` 和 `data/raw/KuaiRand-1K` 使用软链接指向上述原始目录，不复制原始数据。处理后的 Parquet、词表、元数据和缓存写入项目 `data/`。

先用 KuaiRand-1K 完成调试，再用 KuaiRand-27K 运行正式实验。所有数据处理代码参考 UniRank 的 KuaiRand 预处理脚本。允许为 1K/27K 文件名和路径增加配置入口，不改变其特征定义、时间切分和因果历史构造原则。

### 4.1 数据处理原则

- 合并 UniRank 预处理使用的 random 与 standard 日志文件。
- 按时间排序，按日期进行 80% / 10% / 10% 的 train / validation / test 切分。
- 每个预测目标只使用目标时刻之前的历史事件。
- 保留用户历史中的 Item ID、Action 和 Timestamp。
- 只使用目标时刻可获得的特征。
- 不使用 `video_features_statistic` 中按整月汇总的统计字段，避免未来信息泄漏。
- 数值字段按 UniRank 的方式离散化或分桶，类别字段建立词表，ID 0 保留给 OOV / Padding。
- 训练目标先使用单任务 `is_click`。历史 action 仍保留 UniRank 定义的历史反馈编码。
- 训练样本抽样必须发生在时间切分之后。抽样只改变监督目标集合，不得截断构造这些目标所需的真实历史。
- 1K 用于开发和主实验。默认 train / validation / test 目标样本上限分别为 1,000,000 / 100,000 / 100,000；快速调试可配置为 200,000 / 20,000 / 20,000。
- 27K 用于可选规模验证，默认 train / validation / test 目标样本上限分别为 1,000,000 / 100,000 / 100,000。采样按日期分层并固定随机种子。
- 预处理优先采用流式读取和分块写入，禁止一次性将全部 CSV 或完整历史加载到内存。

### 4.2 非序列特征组

为保证类型比较公平，初始实验使用下列 12 个可变字段，每组 4 个。其余 KuaiRand 字段暂不加入本阶段的实验池。

| 类型 | 字段 |
|---|---|
| User Profile | `user_active_degree`, `follow_user_num_range`, `fans_user_num_range`, `register_days_range` |
| Context | `tab`, `day_of_week`, `is_weekend`, `hour` |
| Target Item Metadata | `video_type`, `primary_tag`, `music_type`, `duration_bucket` |

以上字段与 UniRank KuaiRand 预处理中的特征定义保持一致。

OneTrans 中，User Profile 和 Context 通过非 Item 输入进入 Tokenizer。Target Item Metadata 的可变输入只作用于目标物品侧 Tokenizer 输入；历史行为中的 Item Metadata 保持不变，以保证历史序列输入在所有实验条件下相同。目标 Item ID 始终保留。

### 4.3 数据输出

```text
data/processed/KuaiRand-1K/
data/processed/KuaiRand-27K/
```

每个数据集目录保存 train / validation / test Parquet 分块、词表、字段元数据和切分信息。`data/metadata/` 保存字段顺序、质量分数、采样索引和实验数据摘要。

## 5. 公共模型与训练设置

建立一个统一的 OneTrans 实验入口。所有实验使用相同模型骨干、优化器、损失、训练轮数、早停规则和数据切分。实验变量只通过配置传入。

建议初始配置：

| 参数 | 值 |
|---|---:|
| embedding_dim | 16 |
| token_dim | 64 |
| num_layers | 4 |
| num_heads | 4 |
| num_ns_token | 4 |
| tokenizer_type | `Auto` |
| reduction_ratio | 0.0 |
| max_history_len | 100 |
| task | `is_click` |
| loss | Binary Cross Entropy |
| optimizer | 按 UniRank OneTrans 的 dense / sparse 参数分组逻辑独立实现 |

这些参数统一保存在 `config/onetrans.py`。调试时先使用单卡，正式实验支持 4 卡 DDP。随机种子默认 `[2026, 2027, 2028]`。

所有实验保持完整 Tokenizer 输入维度和 `num_ns_token` 不变。通过字段 mask 将未启用字段的 Embedding 输入置零，确保可变的是有效输入字段数量，模型结构和可训练参数规模保持一致。字段顺序固定并写入元数据。

OneTrans 的序列表示诊断必须提取 Backbone 输入和每层输出的 `s_tokens`。序列有效位置以 OneTrans 的历史 mask 为准。

## 6. 实验指标

### 6.1 主指标：Normalized Effective Rank

对某一层收集的有效序列 Token 表示构造矩阵：

\[
H^{(l)}\in\mathbb R^{N\times d}
\]

按列中心化后进行 SVD，得到奇异值 \(\sigma_i\)。定义：

\[
p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
ER^{(l)}=\exp\left(-\sum_i p_i\log p_i\right)
\]

\[
NER^{(l)}=ER^{(l)}/d
\]

报告 Backbone 输入及每层的 NER。所有实验使用完全相同的固定诊断样本和 Token 采样规则，每个配置最多采样 50,000 个有效序列 Token。计算时使用 FP32。

### 6.2 辅助指标

- **Sequence Probe AUC**：冻结 OneTrans，在 train / validation / test 上分别抽取最后一层有效序列 Token。使用 masked mean pooling 得到序列向量，在训练集表示上拟合 Logistic Regression 或单层 Linear Probe，再报告 validation / test AUC。Probe 的结构和训练设置在所有实验中固定。
- **Full CTR AUC**：OneTrans 完整模型在 `is_click` 任务上的 AUC。
- **Full CTR LogLoss**：OneTrans 完整模型在 `is_click` 任务上的 LogLoss。
- **训练成本**：参数量、每秒样本数、峰值显存。只需在主要配置中记录。

NER 是问题诊断主指标。Sequence Probe AUC 用于判断序列表示的任务判别信息是否同步变化。Full CTR AUC / LogLoss 用于观察统一模型的整体预测表现，不要求它们在问题发现实验中必然下降。

## 7. 实验 1.1：非序列特征数量

**问题：** 固定特征质量、类型比例和历史长度，增加可变非序列字段数量是否影响序列表示？

固定 `max_history_len=100`，原始字段值不做扰动，三类字段按 1:1:1 增加。每增加一个字段组，分别从 User、Context、Target Item Metadata 各启用一个字段。

```python
FEATURE_COUNTS = [0, 3, 6, 9, 12]
```

使用固定字段顺序，完整顺序保存在 `data/metadata/`。每个数量条件均保持目标 Item ID 不变。被关闭的字段在 OneTrans Tokenizer 输入前置零，其字段槽位和模型参数保持不变。

输出：

- 各 K 的逐层 NER 曲线。
- 各 K 的 Sequence Probe AUC、Full CTR AUC 和 LogLoss。
- NER 与 K 的关系图。

## 8. 实验 1.2：非序列特征质量

**问题：** 在字段数量和类型固定时，降低非序列特征携带的有效信息是否会影响序列表示？

### 8.1 特征信息量评分

先训练一个仅使用候选非序列字段及目标 Item ID 的 Static Feature Probe。按时间从训练集内部划分 `train_fit` 和 `train_calibration`。只用 `train_fit` 训练 Probe。

对每个字段 \(f_j\)，在 `train_calibration` 上计算置换重要性：

\[
Q_j=\operatorname{LogLoss}(f_j\text{ 被置换})-
\operatorname{LogLoss}(原始输入)
\]

每个字段进行 3 次置换并取平均。\(Q_j\) 越大，表示该字段在当前 Probe 和其他输入条件下具有更高的增量预测价值。该分数只用于描述字段信息量和验证扰动效果，不使用 validation / test 标签计算。

### 8.2 可控质量水平

固定 12 个候选字段全部启用，固定历史长度 100。定义 `q` 为字段值保持真实对应关系的比例：

```python
QUALITY_LEVELS = [1.0, 0.75, 0.5, 0.25, 0.0]
```

其余 `1-q` 比例的字段值通过置换替换，保留该字段的边际取值分布。

- User Profile：以 user_id 为单位置换，保证同一用户在所有目标样本中的字段值一致。
- Target Item Metadata：以 target video_id 为单位置换，保证同一视频的目标侧字段值一致。历史序列中的物品属性不变。
- Context：在各数据 split 内按样本置换，保留该 split 中字段的边际分布。
- 所有置换映射固定随机种子并保存到 `data/metadata/`，不同模型和条件复用相同映射。

另报告 12 个字段的 `Q_j` 排序表，以及各 q 下实际字段值与原始值的一致比例，验证质量水平确实按预期变化。

输出：

- 各 q 的逐层 NER 曲线。
- 各 q 的 Sequence Probe AUC、Full CTR AUC 和 LogLoss。
- `Q_j` 字段重要性表。

## 9. 实验 1.3：非序列特征类型

**问题：** 固定历史长度和字段质量，不同类型的非序列字段是否对序列表示产生不同影响？

固定 `max_history_len=100`，所有启用字段均使用原始值。比较以下条件：

| 条件 | 启用字段 | K |
|---|---|---:|
| User | 4 个 User Profile 字段 | 4 |
| Context | 4 个 Context 字段 | 4 |
| Target Item | 4 个 Target Item Metadata 字段 | 4 |
| All | 三组全部字段 | 12 |
| None | 不启用可变字段，保留目标 Item ID | 0 |

该实验分别报告单一类型的影响及三类特征同时输入的结果。各条件使用同一批目标样本、相同 OneTrans 配置和相同随机种子。

输出：

- 各类型条件的逐层 NER。
- 各类型条件的 Sequence Probe AUC、Full CTR AUC 和 LogLoss。
- 类型对比图。

## 10. 实验 1.4：历史序列长度（可选）

**问题：** 在非序列特征固定时，历史长度是否改变序列表示退化程度？

固定 12 个非序列字段全部启用且不扰动，使用同一批至少拥有 500 条历史行为的目标样本，比较：

```python
HISTORY_LENGTHS = [50, 100, 200, 500]
```

每个条件截取目标时刻之前最近的 L 条历史行为。`reduction_ratio` 始终为 0.0。

输出：

- 各长度的逐层 NER。
- Sequence Probe AUC、Full CTR AUC 和 LogLoss。
- NER 与历史长度的关系图。

若 1K 数据可用的共同目标样本不足，则只在 27K 上运行该实验；不得为不同 L 使用不同用户或不同目标样本集合。

## 11. 结果保存

每次运行必须保存：

```text
data/
├── checkpoints/<experiment>/<seed>/
├── logs/<experiment>/<seed>.log
├── metrics/<experiment>/<seed>.json
├── metrics/summary.csv
└── figures/
```

每份结果 JSON 至少记录：数据集版本、OneTrans 参考仓库 URL 与上游 commit hash、配置快照、随机种子、实验变量、样本数、训练 epoch、最佳 checkpoint、逐层 NER、Probe AUC、Full AUC、LogLoss 和训练成本。

所有结果表保留每个随机种子的原始值，并额外输出 mean / std。图表同时显示均值和标准差。

## 12. 运行命令

README 只保留如下形式的命令和一行说明：

```bash
bash scripts/00_link_raw_data.sh
bash scripts/01_preprocess_kuairand.sh --dataset 1k
bash scripts/11_run_exp_1_1.sh --dataset 1k
bash scripts/12_run_exp_1_2.sh --dataset 1k
bash scripts/13_run_exp_1_3.sh --dataset 1k
bash scripts/14_run_exp_1_4.sh --dataset 1k
bash scripts/20_summarize_results.sh
```

正式实验将 `--dataset 1k` 替换为 `--dataset 27k`。脚本实际参数应与上述命令保持一致。

## 13. 验收标准

- 能从给定原始路径建立软链接并完成 KuaiRand-1K 预处理。
- 能用独立实现的 OneTrans 结构完成单卡训练与验证，且不依赖 UniRank 源码。
- OneTrans 能返回 Backbone 输入及每层序列 Token 表示，正常训练时不额外保存中间激活。
- 所有实验的 OneTrans 结构参数、目标样本、数据切分和训练设置一致。
- 1.1—1.3 可独立运行并自动保存指标和图表。
- 1.4 可通过配置关闭。
- `reduction_ratio=0.0`，诊断时不发生层间序列 Token 数量裁剪。
- 字段数量变化只改变有效字段输入，不改变 Tokenizer 槽位数和模型参数规模。
- 1.2 的字段质量分数、置换映射和实际质量水平可复现。
- README 只包含简短说明和运行命令。
- 本阶段不实现 AFG、Gate、特征选择器或任何 2.1—2.3 解决实验。
