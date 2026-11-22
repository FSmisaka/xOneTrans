"""Deterministic permutation maps for exp 1.2 feature-quality control.

For quality level q, a fraction (1 - q) of field values is permuted while the
marginal value distribution is preserved:
- User Profile fields: permuted at user granularity (a user's value is
  consistent across all of the user's samples).
- Target Item Metadata fields: permuted at video granularity; history items
  are never touched.
- Context fields: permuted per split at sample granularity.

Maps are built once per dataset and reused across all models/conditions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS
from config.dataset import CONFIG as DATA_CFG


def _partial_permutation(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(n)
    if frac <= 0:
        return perm
    sel = rng.choice(n, size=int(round(n * frac)), replace=False)
    perm[sel] = sel[rng.permutation(len(sel))]
    return perm


def _entity_consistency(base: np.ndarray, perm: np.ndarray) -> float:
    return float((base[perm] == base).mean())


def build_quality_maps(dataset_key: str, quality_levels, seed: int, force: bool = False):
    root = PATHS["processed_dataset_dirs"][dataset_key]
    out_dir = PATHS["metadata_root"] / "quality_permutations" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "consistency.json"
    stats = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    for q in quality_levels:
        fp = out_dir / f"q_{q:.2f}.npz"
        if fp.exists() and not force:
            continue
        rng = np.random.default_rng(seed + int(q * 1000))
        split_dfs = {
            "train": pd.read_parquet(root / "train.parquet"),
            "valid": pd.read_parquet(root / "validation.parquet"),
            "test": pd.read_parquet(root / "test.parquet"),
        }
        item_info = pd.read_parquet(root / "item_info.parquet")
        first = split_dfs["train"].drop_duplicates("user_index").set_index("user_index").sort_index()
        n_users = int(split_dfs["train"]["user_index"].max()) + 1
        user_base = np.zeros((n_users, len(DATA_CFG["user_profile_fields"])), dtype=np.int32)
        user_base[first.index.values] = first[DATA_CFG["user_profile_fields"]].values.astype(np.int32)
        n_items = len(item_info)
        item_base = item_info[DATA_CFG["target_item_fields"]].values.astype(np.int32)

        maps = {}
        frac = 1.0 - q
        stats_q = {"q": q, "user_entity": {}, "item_entity": {}, "context_row": {}}
        for j, f in enumerate(DATA_CFG["user_profile_fields"]):
            maps[f"user__{f}"] = _partial_permutation(n_users, frac, rng).astype(np.int32)
            stats_q["user_entity"][f] = _entity_consistency(user_base[:, j], maps[f"user__{f}"])
        for j, f in enumerate(DATA_CFG["target_item_fields"]):
            maps[f"item__{f}"] = _partial_permutation(n_items, frac, rng).astype(np.int32)
            stats_q["item_entity"][f] = _entity_consistency(item_base[:, j], maps[f"item__{f}"])
        for split, df in split_dfs.items():
            for f in DATA_CFG["context_fields"]:
                perm = _partial_permutation(len(df), frac, rng).astype(np.int32)
                maps[f"context__{split}__{f}"] = perm
                stats_q["context_row"].setdefault(split, {})[f] = _entity_consistency(
                    df[f].values.astype(np.int32), perm
                )
        np.savez_compressed(fp, **maps)
        stats[f"{q:.2f}"] = stats_q
        print(f"  built quality maps q={q:.2f} -> {fp}")
    stats_path.write_text(json.dumps(stats, indent=2))
    return out_dir, stats


def load_quality_maps(dataset_key: str, q: float):
    fp = PATHS["metadata_root"] / "quality_permutations" / dataset_key / f"q_{q:.2f}.npz"
    if not fp.exists():
        raise FileNotFoundError(f"quality map not found: {fp}; build maps first")
    raw = np.load(fp)
    maps = {"user": {}, "item": {}, "context": {s: {} for s in ("train", "valid", "test")}}
    for key in raw.files:
        kind, rest = key.split("__", 1)
        if kind == "context":
            split, f = rest.split("__", 1)
            maps["context"][split][f] = raw[key]
        else:
            maps[kind][rest] = raw[key]
    return maps
