"""Aggregate all per-seed metric JSONs into results/metrics/summary.csv."""

import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS


def flatten(metrics: dict) -> dict:
    row = {
        "experiment": metrics["experiment"],
        "condition": metrics["condition"],
        "dataset": metrics["dataset"],
        "seed": metrics["seed"],
        "variable": json.dumps(metrics.get("variable", {})),
        "train_samples": metrics.get("sample_counts", {}).get("train"),
        "valid_samples": metrics.get("sample_counts", {}).get("valid"),
        "test_samples": metrics.get("sample_counts", {}).get("test"),
        "epochs_run": metrics.get("training", {}).get("epochs_run"),
        "best_epoch": metrics.get("training", {}).get("best_epoch"),
        "best_valid_auc": metrics.get("training", {}).get("best_valid_auc"),
        "ner_tokens": metrics.get("ner_tokens"),
        "probe_valid_auc": metrics.get("probe_valid_auc"),
        "probe_test_auc": metrics.get("probe_test_auc"),
        "full_valid_auc": metrics.get("full_valid_auc"),
        "full_valid_logloss": metrics.get("full_valid_logloss"),
        "full_test_auc": metrics.get("full_test_auc"),
        "full_test_logloss": metrics.get("full_test_logloss"),
        "n_params": metrics.get("training_cost", {}).get("n_params"),
        "throughput_samples_per_sec": metrics.get("training_cost", {}).get("throughput_samples_per_sec"),
        "peak_memory_mb": metrics.get("training_cost", {}).get("peak_memory_mb"),
        "shuffle_delta_order_auc": (metrics.get("history_shuffle") or {}).get("delta_order_auc"),
        "shuffle_delta_order_logloss": (metrics.get("history_shuffle") or {}).get("delta_order_logloss"),
        "shuffle_permuted_test_auc": (metrics.get("history_shuffle") or {}).get("permuted_test_auc"),
        "gate_mean": ((metrics.get("gate_stats") or {}).get("overall") or {}).get("mean"),
        "gate_std": ((metrics.get("gate_stats") or {}).get("overall") or {}).get("std"),
    }
    for arm, vals in (metrics.get("gate_intervention") or {}).items():
        row[f"intervention__{arm}__test_auc"] = vals.get("test_auc")
        row[f"intervention__{arm}__test_logloss"] = vals.get("test_logloss")
    exp4_names = ("exp_4_continuous_spectral", "exp_4_2")
    if metrics.get("experiment") not in exp4_names:
        for layer, val in (metrics.get("ner") or {}).items():
            row[f"ner__{layer}"] = val
    for split in ("valid", "test"):
        for path, vals in (metrics.get(split) or {}).items():
            row[f"{split}__{path}__auc"] = vals.get("auc")
            row[f"{split}__{path}__logloss"] = vals.get("logloss")
    for ner_key in ("ner_valid", "ner_test"):
        for branch, layers in (metrics.get(ner_key) or {}).items():
            for layer, val in layers.items():
                row[f"{ner_key}__{branch}__{layer}"] = val
    if metrics.get("mode"):
        row["mode"] = metrics["mode"]
        row["lambda_high"] = (metrics.get("variable") or {}).get("lambda_high")
        row["lambda_low"] = (metrics.get("variable") or {}).get("lambda_low")
    if metrics.get("experiment") in exp4_names:
        row["kind"] = metrics.get("kind")
        row["fusion"] = metrics.get("fusion")
        row["alpha"] = metrics.get("alpha")
        row["node_aux_loss_weight"] = metrics.get("node_aux_loss_weight")
        for split in ("valid", "test"):
            primary = metrics.get(f"{split}_primary") or {}
            row[f"{split}_primary_auc"] = primary.get("auc")
            row[f"{split}_primary_logloss"] = primary.get("logloss")
            for alpha, vals in (metrics.get("node_metrics", {}).get(split) or {}).items():
                row[f"{split}_node_{alpha}__auc"] = vals.get("auc")
                row[f"{split}_node_{alpha}__logloss"] = vals.get("logloss")
        summary = metrics.get("model_summary") or {}
        row["n_params_total"] = summary.get("n_params_total")
        row["n_params_trainable"] = summary.get("n_params_trainable")
        row["train_seconds"] = summary.get("train_seconds")
    return row


def main():
    rows = []
    for fp in sorted(PATHS["metrics_root"].glob("*/*.json")):
        if fp.name == "summary.json":
            continue
        rows.append(flatten(json.loads(fp.read_text())))
    if not rows:
        print("no per-seed metric files found")
        return
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("experiment", "condition", "dataset", "seed"), k))
    out = PATHS["metrics_root"] / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
