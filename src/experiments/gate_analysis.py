"""Analyses for exp 2.x: gate statistics, history shuffling sensitivity and
gate intervention metrics (SPEC2 sections 6.2-6.4)."""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.dataset import CONFIG as DATA_CFG
from src.data.dataset import KuaiRandSequenceDataset, collate_batch
from src.experiments.diagnostics import evaluate
from src.utils.metrics import auc_score, logloss_score


def history_shuffle_metrics(model, data, max_len: int, seed: int, batch_size: int,
                            device, orig_auc: float, orig_logloss: float,
                            max_rows: int = None) -> Dict:
    """Test-time history-order shuffling: same items/actions/target/labels,
    valid positions permuted, padding untouched (SPEC2 6.2)."""
    ds = KuaiRandSequenceDataset(data, "test", max_len, shuffle_history_seed=seed,
                                 max_rows=max_rows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_batch, num_workers=0)
    res = evaluate(model, loader, device)
    perm_auc = auc_score(res["y_true"], res["y_pred"])
    perm_ll = logloss_score(res["y_true"], res["y_pred"])
    return {
        "shuffle_seed": seed,
        "orig_test_auc": orig_auc,
        "permuted_test_auc": perm_auc,
        "delta_order_auc": orig_auc - perm_auc,
        "orig_test_logloss": orig_logloss,
        "permuted_test_logloss": perm_ll,
        "delta_order_logloss": perm_ll - orig_logloss,
        "n_samples": int(len(res["y_true"])),
    }


@torch.no_grad()
def _collect_gate_values(model, dataset: KuaiRandSequenceDataset, batch_size: int, device):
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_batch, num_workers=0)
    gates, preds, labels = [], [], []
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev, return_gate=True)
        gates.append(out["gate"].float().cpu().numpy())
        preds.append(out["pred"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(gates), np.concatenate(preds), np.concatenate(labels)


def _group_metrics(y, p, g, mask):
    y, p, g = y[mask], p[mask], g[mask]
    entry = {"n": int(mask.sum()), "gate_mean": float(g.mean())}
    if len(np.unique(y)) > 1:
        entry["auc"] = auc_score(y, p)
        entry["logloss"] = logloss_score(y, p)
    else:
        entry["auc"] = float("nan")
        entry["logloss"] = float("nan")
    return entry


def collect_gate_statistics(model, dataset: KuaiRandSequenceDataset, batch_size: int,
                            device, length_buckets: List[int], group_field: str,
                            out_dir: Path, seed: int) -> Dict:
    """Gate value distribution, per-dim means, buckets by history length and by
    a target item metadata field (SPEC2 6.3)."""
    g, p, y = _collect_gate_values(model, dataset, batch_size, device)
    lengths = np.minimum(dataset.position, dataset.max_len)
    group_vals = dataset.item_cols[group_field] if group_field in dataset.item_cols \
        else np.zeros(len(dataset), dtype=np.int64)
    flat = g.flatten()
    stats = {
        "n_samples": int(g.shape[0]),
        "overall": {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            **{f"p{q}": float(np.percentile(flat, q)) for q in (10, 25, 50, 75, 90)},
        },
        "per_dim_mean": g.mean(axis=0).tolist(),
        "by_history_len": [],
        "by_group_field": group_field,
        "by_group": [],
    }
    for lo, hi in zip(length_buckets[:-1], length_buckets[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        if mask.any():
            stats["by_history_len"].append(
                {"bucket": f"[{lo},{hi})", **_group_metrics(y, p, g.mean(axis=1), mask)})
    for v in np.unique(group_vals):
        mask = group_vals == v
        if mask.any():
            stats["by_group"].append({"group": int(v), **_group_metrics(y, p, g.mean(axis=1), mask)})

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"gate_values_{seed}.npy", g.astype(np.float32))
    with open(out_dir / f"gate_stats_{seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "key", "n", "gate_mean", "auc", "logloss"])
        w.writeheader()
        for row in stats["by_history_len"]:
            w.writerow({"scope": "history_len", "key": row["bucket"], "n": row["n"],
                        "gate_mean": row["gate_mean"], "auc": row["auc"], "logloss": row["logloss"]})
        for row in stats["by_group"]:
            w.writerow({"scope": stats["by_group_field"], "key": row["group"], "n": row["n"],
                        "gate_mean": row["gate_mean"], "auc": row["auc"], "logloss": row["logloss"]})
    return stats
