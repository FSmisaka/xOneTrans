# xOneTrans

OneTrans 序列表示退化诊断实验（问题发现阶段，实验 1.1–1.4）。

## 环境安装

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt  # 安装依赖（torch、pandas、pyarrow、scikit-learn、matplotlib）
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
bash scripts/20_summarize_results.sh          # 汇总全部结果到 data/metrics/summary.csv
```

正式实验将 `--dataset 1k` 替换为 `--dataset 27k`。所有脚本支持 `--seeds`、`--device` 透传，例如 `bash scripts/11_run_exp_1_1.sh --dataset 1k --seeds 2026`。

## 多卡训练（可选）

```bash
torchrun --nproc_per_node=4 -m src.experiments.run_experiment --exp 1_1 --dataset 27k  # 4 卡 DDP 训练
```

## 结果位置

- 指标：`data/metrics/<experiment>/<seed>.json` 与 `data/metrics/summary.csv`
- 图表：`data/figures/`
- 检查点：`data/checkpoints/<experiment>/<seed>/best.pt`
- 日志：`data/logs/<experiment>/<seed>.log`
- 元数据（字段顺序、质量置换映射、字段重要性、UniRank 参考）：`data/metadata/`
