"""Independent KuaiRand preprocessing for OneTrans diagnostics.

Follows the UniRank reference (data/KuaiRand/preprocess_Kuairand_seq_action.py)
in field definitions, 64-pattern action encoding, date-based 8:1:1 split and
causal history construction, adapted to this project's paths and 1K/27K
filenames. Streaming: logs are read in CSV chunks, spilled to user-hash
partitions, then processed partition by partition.
"""

import argparse
import gc
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS
from config.dataset import CONFIG as DATA_CFG


def build_action_codes(label_columns):
    pat2name = {}
    for p in range(64):
        pat2name[p] = "exposure" if p == 0 else "|".join(
            c for i, c in enumerate(label_columns) if p & (1 << i)
        )
    name2code = {n: i + 1 for i, n in enumerate(sorted(set(pat2name.values())))}
    return np.array([name2code[pat2name[p]] for p in range(64)], dtype=np.int32)


def build_date_split(sorted_dates, ratios):
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError("need at least 3 distinct dates for a date-based split")
    r = np.array([ratios["train"], ratios["valid"], ratios["test"]], dtype=np.float64)
    r = r / r.sum()
    raw = r * n_days
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts))
    for i in range(n_days - counts.sum()):
        counts[order[i % 3]] += 1
    for idx in range(3):
        while counts[idx] == 0 and counts.max() > 1:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[idx] += 1
    n_train, n_valid = int(counts[0]), int(counts[1])
    return {
        "n_days": n_days,
        "train_dates": sorted_dates[:n_train],
        "valid_dates": sorted_dates[n_train:n_train + n_valid],
        "test_dates": sorted_dates[n_train + n_valid:],
        "valid_start_date": sorted_dates[n_train],
        "test_start_date": sorted_dates[n_train + n_valid],
    }


def fmt_date(d):
    return f"{d // 10000:04d}-{d % 10000 // 100:02d}-{d % 100:02d}"


class PartitionSpiller:
    def __init__(self, interim_dir: Path, n_parts: int, flush_rows: int):
        self.interim_dir = interim_dir
        self.n_parts = n_parts
        self.flush_rows = flush_rows
        self.buffers = [[] for _ in range(n_parts)]
        self.sizes = [0] * n_parts
        self.file_counts = [0] * n_parts
        for p in range(n_parts):
            (interim_dir / f"part_{p:03d}").mkdir(parents=True, exist_ok=True)

    def add(self, part_id: int, df: pd.DataFrame):
        self.buffers[part_id].append(df)
        self.sizes[part_id] += len(df)
        if self.sizes[part_id] >= self.flush_rows:
            self.flush(part_id)

    def flush(self, part_id: int):
        if not self.buffers[part_id]:
            return
        out = pd.concat(self.buffers[part_id], ignore_index=True)
        fp = self.interim_dir / f"part_{part_id:03d}" / f"c_{self.file_counts[part_id]:04d}.parquet"
        out.to_parquet(fp, index=False)
        self.file_counts[part_id] += 1
        self.buffers[part_id] = []
        self.sizes[part_id] = 0
        del out

    def flush_all(self):
        for p in range(self.n_parts):
            self.flush(p)

    def partition_files(self, part_id: int):
        return sorted((self.interim_dir / f"part_{part_id:03d}").glob("*.parquet"))


def phase1_stream_logs(raw_dir: Path, interim_dir: Path, log_files, n_parts: int):
    tz = DATA_CFG["timezone"]
    spiller = PartitionSpiller(interim_dir, n_parts, DATA_CFG["partition_flush_rows"])
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    date_counts = defaultdict(int)
    total = 0
    for fname in log_files:
        print(f"  [phase1] {fname}")
        reader = pd.read_csv(
            raw_dir / fname,
            usecols=DATA_CFG["log_columns"],
            chunksize=DATA_CFG["csv_chunk_size"],
        )
        for chunk in reader:
            chunk = chunk.dropna(subset=["user_id", "video_id"])
            dt = pd.to_datetime(chunk["time_ms"], unit="ms", utc=True).dt.tz_convert(tz)
            chunk["day_of_week"] = (dt.dt.dayofweek + 1).astype("int16")
            chunk["is_weekend"] = ((dt.dt.dayofweek >= 5).astype(int) + 1).astype("int16")
            chunk["hour"] = (dt.dt.hour + 1).astype("int16")
            chunk["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("int32")
            chunk["tab"] = ((chunk["tab"] + 1).clip(0, 15)).fillna(0).astype("int16")
            chunk["user_id"] = chunk["user_id"].astype("int64")
            chunk["video_id"] = chunk["video_id"].astype("int64")
            chunk["time_ms"] = chunk["time_ms"].astype("int64")
            for uid, cnt in chunk["user_id"].value_counts(sort=False).items():
                user_counts[int(uid)] += int(cnt)
            for vid, cnt in chunk["video_id"].value_counts(sort=False).items():
                item_counts[int(vid)] += int(cnt)
            for d, cnt in chunk["date"].value_counts(sort=False).items():
                date_counts[int(d)] += int(cnt)
            part_arr = chunk["user_id"].values % n_parts
            for pid in np.unique(part_arr):
                spiller.add(int(pid), chunk.loc[part_arr == pid].copy())
            total += len(chunk)
            del chunk, part_arr, dt
            gc.collect()
        print(f"    accumulated {total:,} rows")
    spiller.flush_all()
    print(f"  [phase1] done: {total:,} rows, {len(user_counts):,} users, "
          f"{len(item_counts):,} items, {len(date_counts)} dates")
    return spiller, user_counts, item_counts, date_counts


def build_vocab_weighted(values, weights, min_count):
    agg = pd.DataFrame({"v": values, "w": weights}).groupby("v")["w"].sum()
    vocab = {"__MISSING__": 1}
    idx = 2
    for v in sorted(agg.index):
        if v in ("nan", "", "__MISSING__"):
            continue
        if agg[v] >= min_count:
            vocab[str(v)] = idx
            idx += 1
    return vocab


def load_user_feature_table(raw_dir: Path, user_features_file: str, user_fields, user_counts, user_idx_map, min_count):
    uf = pd.read_csv(raw_dir / user_features_file, usecols=["user_id"] + user_fields)
    uf = uf[uf["user_id"].isin(user_idx_map)].copy()
    uf["user_index"] = uf["user_id"].map(user_idx_map).astype("int32")
    weights = uf["user_id"].map(user_counts).fillna(0).astype(np.int64).values
    vocabs = {col: build_vocab_weighted(uf[col].astype(str).values, weights, min_count) for col in user_fields}
    uf = uf.sort_values("user_index").reset_index(drop=True)
    table = {"index": uf["user_index"].values.astype(np.int64)}
    for col in user_fields:
        table[col] = uf[col].astype(str).map(vocabs[col]).fillna(0).astype("int32").values
    return table, vocabs


def load_item_feature_table(raw_dir: Path, video_features_file: str, item_counts, item_idx_map, min_count):
    bins = DATA_CFG["duration_bins"]
    labels = DATA_CFG["duration_labels"]
    cols = ["video_id", "video_type", "music_type", "tag", "video_duration"]
    pieces = []
    for chunk in pd.read_csv(raw_dir / video_features_file, usecols=cols, chunksize=DATA_CFG["csv_chunk_size"]):
        pieces.append(chunk[chunk["video_id"].isin(item_counts)])
    vf = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=cols)
    del pieces
    vf["video_id"] = vf["video_id"].astype("int64")
    vf["video_type"] = vf["video_type"].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    vf["music_type"] = vf["music_type"].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    vf["primary_tag"] = vf["tag"].astype(str).str.split(",").str[0].str.strip()
    vf["primary_tag"] = vf["primary_tag"].replace({"nan": "", "": "__MISSING__"})
    vf["duration_bucket"] = pd.cut(
        pd.to_numeric(vf["video_duration"], errors="coerce").fillna(0),
        bins=bins, labels=labels,
    ).astype(str)
    vf["duration_bucket"] = vf["duration_bucket"].replace({"nan": "__MISSING__"})
    vf = vf.drop_duplicates(subset=["video_id"], keep="last")
    weights = vf["video_id"].map(item_counts).fillna(0).astype(np.int64).values
    item_fields = DATA_CFG["target_item_fields"]
    vocabs = {col: build_vocab_weighted(vf[col].values, weights, min_count) for col in item_fields}
    vf["item_index"] = vf["video_id"].map(item_idx_map).astype("int32")
    vf = vf.sort_values("item_index").reset_index(drop=True)
    table = {"index": vf["item_index"].values.astype(np.int64)}
    for col in item_fields:
        table[col] = vf[col].astype(str).map(vocabs[col]).fillna(0).astype("int32").values
    return table, vocabs


def codes_from_table(table, keys: np.ndarray, field: str) -> np.ndarray:
    idx = table["index"]
    pos = np.searchsorted(idx, keys)
    pos = np.clip(pos, 0, len(idx) - 1)
    hit = idx[pos] == keys
    return np.where(hit, table[field][pos], 0).astype("int32")


def process_partition(spiller, pid, user_idx_map, item_idx_map, user_table, item_table,
                      action_codes, valid_start, test_start, keep_ratios, sampling_seed):
    files = spiller.partition_files(pid)
    if not files:
        return [], {"train": [], "valid": [], "test": []}
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    gc.collect()

    label_cols = DATA_CFG["label_columns"]
    binary = df[label_cols].fillna(0).astype(np.int8).values
    pattern = np.zeros(len(df), dtype=np.int64)
    for i in range(len(label_cols)):
        pattern |= binary[:, i].astype(np.int64) << i
    df["action"] = action_codes[pattern].astype("int32")

    df = df.sort_values(["user_id", "time_ms"], kind="stable").reset_index(drop=True)
    df["position"] = df.groupby("user_id", sort=False).cumcount().astype("int32")
    df["user_index"] = df["user_id"].map(user_idx_map)
    df["item_index"] = df["video_id"].map(item_idx_map).astype("int32")
    df = df.dropna(subset=["user_index"]).reset_index(drop=True)
    df["user_index"] = df["user_index"].astype("int32")
    df["is_click"] = df["is_click"].fillna(0).astype(np.float32)
    for col in DATA_CFG["context_fields"]:
        df[col] = df[col].fillna(0).astype("int32")

    user_info_rows = []
    for _, g in df.groupby("user_index", sort=True):
        user_info_rows.append({
            "user_index": int(g["user_index"].iloc[0]),
            "item_seq": g["item_index"].astype(np.int32).tolist(),
            "action_seq": g["action"].astype(np.int32).tolist(),
            "timestamp_seq": g["time_ms"].astype(np.int64).tolist(),
        })

    date = df["date"].values
    is_test = date >= test_start
    is_valid = (date >= valid_start) & ~is_test
    split_of = np.where(is_test, 2, np.where(is_valid, 1, 0))

    kept = {"train": [], "valid": [], "test": []}
    for s, split in enumerate(["train", "valid", "test"]):
        mask = split_of == s
        if not mask.any():
            continue
        sdf = df.loc[mask]
        rng = np.random.default_rng(sampling_seed + pid * 10 + s)
        if keep_ratios[split] < 1.0:
            sdf = sdf.loc[rng.random(len(sdf)) < keep_ratios[split]]
        if len(sdf) == 0:
            continue
        u_idx = sdf["user_index"].values.astype(np.int64)
        i_idx = sdf["item_index"].values.astype(np.int64)
        out = pd.DataFrame({
            "user_index": sdf["user_index"].astype("int32"),
            "item_index": sdf["item_index"].astype("int32"),
            "position": sdf["position"].astype("int32"),
            "is_click": sdf["is_click"].astype(np.float32),
        })
        for col in DATA_CFG["user_profile_fields"]:
            out[col] = codes_from_table(user_table, u_idx, col)
        for col in DATA_CFG["context_fields"]:
            out[col] = sdf[col].astype("int32").values
        for col in DATA_CFG["target_item_fields"]:
            out[col] = codes_from_table(item_table, i_idx, col)
        out["date"] = sdf["date"].astype("int32")
        kept[split].append(out)
        del sdf, out
    del df
    gc.collect()
    return user_info_rows, kept


def preprocess(dataset_key: str, debug: bool):
    raw_dir = PATHS["raw_dataset_dirs"][dataset_key] / "data"
    out_dir = PATHS["processed_dataset_dirs"][dataset_key]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    interim_dir = PATHS["interim_root"] / f"kuairand_{dataset_key}"
    if interim_dir.exists():
        shutil.rmtree(interim_dir)
    interim_dir.mkdir(parents=True)

    caps = DATA_CFG["debug_sample_caps"] if debug else DATA_CFG["sample_caps"]
    seed = DATA_CFG["sampling_seed"]
    n_parts = DATA_CFG["n_user_partitions"][dataset_key]
    user_fields = DATA_CFG["user_profile_fields"]
    item_fields = DATA_CFG["target_item_fields"]
    action_codes = build_action_codes(DATA_CFG["label_columns"])

    print(f"[step1] stream logs -> user-hash partitions (raw={raw_dir})")
    spiller, user_counts, item_counts, date_counts = phase1_stream_logs(
        raw_dir, interim_dir, DATA_CFG["log_files"][dataset_key], n_parts
    )

    print("[step2] date split 8:1:1 by day count")
    sorted_dates = sorted(date_counts)
    split_info = build_date_split(sorted_dates, DATA_CFG["split_ratios"])
    print(f"  {split_info['n_days']} days | train {fmt_date(sorted_dates[0])}~{fmt_date(split_info['train_dates'][-1])}"
          f" | valid from {fmt_date(split_info['valid_start_date'])} | test from {fmt_date(split_info['test_start_date'])}")

    print("[step3] vocabularies + id maps")
    min_inter = DATA_CFG["min_user_interactions"]
    valid_users = sorted(u for u, c in user_counts.items() if c >= min_inter)
    user_idx_map = {u: i for i, u in enumerate(valid_users)}
    item_idx_map = {v: i + 1 for i, v in enumerate(sorted(item_counts))}
    n_users, n_items = len(valid_users), len(item_idx_map)
    print(f"  users={n_users:,} (>={min_inter} interactions), items={n_items:,}")

    user_table, user_vocabs = load_user_feature_table(
        raw_dir, DATA_CFG["user_features_file"][dataset_key], user_fields, user_counts, user_idx_map,
        DATA_CFG["min_feat_count"],
    )
    item_table, item_vocabs = load_item_feature_table(
        raw_dir, DATA_CFG["video_features_file"][dataset_key], item_counts, item_idx_map,
        DATA_CFG["min_feat_count"],
    )

    split_rows = {"train": 0, "valid": 0, "test": 0}
    for d, c in date_counts.items():
        split = "train" if d < split_info["valid_start_date"] else ("valid" if d < split_info["test_start_date"] else "test")
        split_rows[split] += c
    keep_ratios = {s: min(1.0, caps[s] / split_rows[s]) if split_rows[s] else 1.0 for s in caps}
    print(f"  target rows before sampling={split_rows}, keep_ratios={ {k: round(v, 4) for k, v in keep_ratios.items()} }")

    print("[step4] per-partition: causal history + date split + post-split sampling")
    kept_all = {"train": [], "valid": [], "test": []}
    ui_writer = None
    max_seq = 0
    for pid in range(n_parts):
        rows, kept = process_partition(
            spiller, pid, user_idx_map, item_idx_map, user_table, item_table, action_codes,
            split_info["valid_start_date"], split_info["test_start_date"], keep_ratios, seed,
        )
        for s in kept_all:
            kept_all[s].extend(kept[s])
        if rows:
            max_seq = max(max_seq, max(len(r["item_seq"]) for r in rows))
            t = pa.Table.from_pandas(pd.DataFrame(rows))
            if ui_writer is None:
                ui_writer = pq.ParquetWriter(out_dir / "user_info.parquet", t.schema)
            ui_writer.write_table(t)
        n_kept = sum(len(x) for lst in kept.values() for x in lst)
        print(f"  [phase2] partition {pid:03d}: users={len(rows)}, kept_targets={n_kept:,}")
        del rows, kept
        gc.collect()
    if ui_writer is not None:
        ui_writer.close()

    print("[step5] finalize split parquet files")
    split_files = {"train": "train.parquet", "valid": "validation.parquet", "test": "test.parquet"}
    summary_counts = {}
    for split, fname in split_files.items():
        df = pd.concat(kept_all[split], ignore_index=True) if kept_all[split] else pd.DataFrame()
        if len(df) > caps[split]:
            rng = np.random.default_rng(seed)
            df = df.iloc[np.sort(rng.choice(len(df), size=caps[split], replace=False))].reset_index(drop=True)
        df.to_parquet(out_dir / fname, index=False)
        summary_counts[split] = int(len(df))
        print(f"  {split}: {len(df):,} targets -> {fname}")
        del df
    del kept_all
    gc.collect()

    print("[step6] item_info + vocab + metadata")
    item_info = pd.DataFrame({"item_index": np.arange(n_items + 1, dtype=np.int32)})
    for col in item_fields:
        arr = np.zeros(n_items + 1, dtype=np.int32)
        arr[item_table["index"]] = item_table[col]
        item_info[col] = arr
    item_info.to_parquet(out_dir / "item_info.parquet", index=False)

    vocab_json = {
        "n_users": n_users,
        "n_items": n_items,
        "num_action_codes": int(action_codes.max()) + 1,
        "user_vocabs": user_vocabs,
        "item_vocabs": item_vocabs,
        "context_vocabs": {"tab": 16, "day_of_week": 8, "is_weekend": 3, "hour": 25},
    }
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab_json, f, indent=2)
    with open(out_dir / "split_info.json", "w") as f:
        json.dump({
            "split_ratios": DATA_CFG["split_ratios"],
            "n_days": split_info["n_days"],
            "train_dates": [fmt_date(d) for d in split_info["train_dates"]],
            "valid_dates": [fmt_date(d) for d in split_info["valid_dates"]],
            "test_dates": [fmt_date(d) for d in split_info["test_dates"]],
        }, f, indent=2)

    meta = {
        "dataset": DATA_CFG["dataset_names"].get(dataset_key, dataset_key),
        "raw_dir": str(raw_dir),
        "processed_dir": str(out_dir),
        "debug": debug,
        "sample_caps": caps,
        "sampling_seed": seed,
        "min_user_interactions": min_inter,
        "min_feat_count": DATA_CFG["min_feat_count"],
        "total_rows": int(sum(date_counts.values())),
        "split_target_counts": summary_counts,
        "n_users": n_users,
        "n_items": n_items,
        "max_user_seq_len": int(max_seq),
    }
    PATHS["metadata_root"].mkdir(parents=True, exist_ok=True)
    with open(PATHS["metadata_root"] / f"processed_{dataset_key}.json", "w") as f:
        json.dump(meta, f, indent=2)

    shutil.rmtree(interim_dir)
    print(f"[done] processed data at {out_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["1k", "27k"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    preprocess(args.dataset, args.debug)


if __name__ == "__main__":
    main()
