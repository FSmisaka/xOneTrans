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
    }
    for layer, val in (metrics.get("ner") or {}).items():
        row[f"ner__{layer}"] = val
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
