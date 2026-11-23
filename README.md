# xOneTrans

OneTrans 序列表示退化诊断实验（问题发现阶段，实验 1.1–1.4）。

## 环境安装

```bash
source /thuir/wangyiyao/miniconda3/etc/profile.d/conda.sh
conda activate /thuir/wangyiyao/conda_envs/r4gr
python3 -m venv .venv
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt  
# 安装依赖（torch、pandas、pyarrow、scikit-learn、matplotlib）
```

## 数据准备

```bash
bash scripts/00_link_raw_data.sh              # 建立 data/raw 下 KuaiRand-1K/27K 原始数据软链接
bash scripts/01_preprocess_kuairand.sh --dataset 1k   # 预处理 KuaiRand-1K（追加 --debug 使用小样本上限）
bash scripts/01_preprocess_kuairand.sh --dataset 27k  # 预处理 KuaiRand-27K
```

## 运行实验

```bash
bash scripts/11_run_exp_1_1.sh --dataset 1k   # 实验 1.1：非序列特征数量（K=0/3/6/9/12）
bash scripts/12_run_exp_1_2.sh --dataset 1k   # 实验 1.2：非序列特征质量（q=1.0~0.0，含字段重要性 Q_j）
bash scripts/13_run_exp_1_3.sh --dataset 1k   # 实验 1.3：非序列特征类型（User/Context/TargetItem/All/None）
bash scripts/14_run_exp_1_4.sh --dataset 1k   # 实验 1.4：历史长度（可选，默认在 config/exp_1_4_length.py 中关闭）
bash scripts/21_run_exp_2_1_task_aware_gate.sh --dataset 1k   # 实验 2.1：Task-aware Residual Gate vs Baseline
bash scripts/22_run_exp_2_2_gate_ablation.sh --dataset 1k     # 实验 2.2：Gate 机制消融（Ungated/Fixed/Learned）
bash scripts/23_run_exp_2_3_gate_analysis.sh --dataset 1k     # 实验 2.3：Gate 分布/干预与历史打乱敏感性分析
bash scripts/20_summarize_results.sh          # 汇总全部结果到 results/metrics/summary.csv
```

正式实验将 `--dataset 1k` 替换为 `--dataset 27k`。所有脚本支持 `--seeds`、`--device` 透传，例如 `bash scripts/11_run_exp_1_1.sh --dataset 1k --seeds 2026`。

## 多卡训练（可选）

卡数不在代码中写死，由 torchrun 的 `--nproc_per_node` 指定（2/3/4 卡均可），共享服务器用 `CUDA_VISIBLE_DEVICES` 挑空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 -m src.experiments.run_experiment --exp 1_1 --dataset 1k  # 用指定 4 卡跑实验 1.1
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29501 -m src.experiments.run_experiment --exp 1_2 --dataset 1k  # 同时用另外 2 张卡跑实验 1.2（注意换 master_port）
```

有效 batch（config/onetrans.py 的 batch_size=2048）在任意卡数下保持不变：micro batch = batch_size ÷ 累积步数 ÷ 卡数，全局梯度一致。

## 结果位置

- 指标：`results/metrics/<experiment>/<seed>.json` 与 `results/metrics/summary.csv`
- 图表：`results/figures/`
- 检查点：`results/checkpoints/<experiment>/<seed>/best.pt`
- 日志：`results/logs/<experiment>/<seed>.log`
- 数据侧元数据（字段顺序、质量置换映射、字段重要性、UniRank 参考）：`data/metadata/`（属于实验数据定义，随数据而非模型输出）
