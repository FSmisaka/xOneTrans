"""Result figures. All plots show mean and std across seeds."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _style():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def plot_ner_curves(ner_by_cond, title, out_path: Path):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, layers in ner_by_cond.items():
        names = list(layers.keys())
        means = np.array([layers[n]["mean"] for n in names])
        stds = np.array([layers[n]["std"] for n in names])
        x = np.arange(len(names))
        ax.plot(x, means, marker="o", label=str(cond))
        ax.fill_between(x, means - stds, means + stds, alpha=0.15)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30)
    ax.set_xlabel("layer")
    ax.set_ylabel("Normalized Effective Rank")
    ax.set_title(f"{title}: per-layer NER (mean ± std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_ner_vs_x(ner_by_cond, x_vals, xlabel, out_path: Path):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    pairs = sorted(zip(x_vals, list(ner_by_cond.values())))
    for layer_sel, label in [(-1, "last layer"), (0, "backbone input")]:
        xs = [p[0] for p in pairs]
        layer_name = list(pairs[0][1].keys())[layer_sel]
        means = np.array([p[1][layer_name]["mean"] for p in pairs])
        stds = np.array([p[1][layer_name]["std"] for p in pairs])
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=f"NER ({label})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Effective Rank")
    ax.set_title(f"NER vs {xlabel} (mean ± std)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_field_importance(q_j: dict, out_path: Path):
    _style()
    items = sorted(q_j.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_xlabel("Q_j (permutation importance, LogLoss increase)")
    ax.set_title("Static Feature Probe: field information scores")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_group_bars(metric_by_cond, title, out_path: Path, ylabel: str = "value"):
    _style()
    conds = list(metric_by_cond.keys())
    keys = [k for k in metric_by_cond[conds[0]]]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    axes = np.atleast_1d(axes)
    x = np.arange(len(conds))
    for i, k in enumerate(keys):
        means = [metric_by_cond[c][k]["mean"] for c in conds]
        stds = [metric_by_cond[c][k]["std"] for c in conds]
        axes[i].bar(x, means, yerr=stds, capsize=3)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(conds, rotation=20)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(k)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_gate_outputs(experiment_summary: dict, figures_root: Path, exp_id: str):
    """Gate distribution / gate-vs-length / gate-vs-group-AUC plots for exp 2.3."""
    import json as _json
    _style()
    metrics_root = figures_root.parent / "metrics"
    gate_npy_files = sorted(metrics_root.glob(f"{exp_id}_*/gate_values_*.npy"))
    if gate_npy_files:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 1, 51)
        for fp in gate_npy_files:
            g = np.load(fp).flatten()
            ax.hist(g, bins=bins, density=True, alpha=0.4, label=fp.parent.name.split("_")[-1] + "/" + fp.stem.split("_")[-1])
        ax.set_xlabel("gate value")
        ax.set_ylabel("density")
        ax.set_title("Gate value distribution (test set, per seed)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_root / f"{exp_id}_gate_distribution.png")
        plt.close(fig)

    csvs = sorted(metrics_root.glob(f"{exp_id}_*/gate_stats_*.csv"))
    if csvs:
        rows = []
        for fp in csvs:
            with open(fp) as f:
                for r in csv.DictReader(f):
                    r["seed"] = fp.stem.split("_")[-1]
                    rows.append(r)
        for scope in sorted({r["scope"] for r in rows}):
            sel = [r for r in rows if r["scope"] == scope]
            keys = sorted({r["key"] for r in sel}, key=lambda k: float(str(k).strip("[]").split(",")[0]))
            fig, ax1 = plt.subplots(figsize=(7, 4))
            xs = np.arange(len(keys))
            gm = [np.nanmean([float(r["gate_mean"]) for r in sel if r["key"] == k]) for k in keys]
            gs = [np.nanstd([float(r["gate_mean"]) for r in sel if r["key"] == k]) for k in keys]
            auc_m = [np.nanmean([float(r["auc"]) for r in sel if r["key"] == k]) for k in keys]
            ax1.bar(xs, gm, yerr=gs, capsize=3, color="tab:blue", alpha=0.7, label="gate mean")
            ax1.set_ylabel("gate mean")
            ax1.set_xticks(xs)
            ax1.set_xticklabels(keys, rotation=30)
            ax2 = ax1.twinx()
            ax2.plot(xs, auc_m, marker="o", color="tab:red", label="bucket AUC")
            ax2.set_ylabel("AUC")
            ax1.set_title(f"Gate vs {scope} (bars: gate, line: AUC)")
            fig.tight_layout()
            fig.savefig(figures_root / f"{exp_id}_gate_vs_{scope}.png")
            plt.close(fig)


def plot_metric_bars(metric_by_cond, title, out_path: Path):
    _style()
    conds = list(metric_by_cond.keys())
    auc_keys = [k for k in ("probe_valid_auc", "probe_test_auc", "full_valid_auc", "full_test_auc")
                if k in metric_by_cond[conds[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(conds))
    width = 0.8 / max(len(auc_keys), 1)
    for i, k in enumerate(auc_keys):
        means = [metric_by_cond[c][k]["mean"] for c in conds]
        stds = [metric_by_cond[c][k]["std"] for c in conds]
        axes[0].bar(x + (i - len(auc_keys) / 2) * width, means, width, yerr=stds, capsize=2, label=k)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds, rotation=20)
    axes[0].set_ylabel("AUC")
    axes[0].legend(fontsize=8)
    axes[0].set_title("AUC metrics")
    if "full_test_logloss" in metric_by_cond[conds[0]]:
        means = [metric_by_cond[c]["full_test_logloss"]["mean"] for c in conds]
        stds = [metric_by_cond[c]["full_test_logloss"]["std"] for c in conds]
        axes[1].bar(x, means, yerr=stds, capsize=3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conds, rotation=20)
        axes[1].set_ylabel("LogLoss")
        axes[1].set_title("Full CTR LogLoss (test)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
