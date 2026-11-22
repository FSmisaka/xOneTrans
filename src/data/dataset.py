"""Processed-data loading, sequence dataset and collation."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS
from config.dataset import CONFIG as DATA_CFG

VARIABLE_FIELDS = (DATA_CFG["user_profile_fields"]
                   + DATA_CFG["context_fields"]
                   + DATA_CFG["target_item_fields"])


class UserHistoryStore:
    """CSR-style storage of full per-user histories (items, actions)."""

    def __init__(self, user_info_path: Path):
        df = pd.read_parquet(user_info_path, columns=["user_index", "item_seq", "action_seq"])
        n_users = int(df["user_index"].max()) + 1
        offsets = np.zeros(n_users + 1, dtype=np.int64)
        for u, items in zip(df["user_index"].values, df["item_seq"].values):
            offsets[int(u) + 1] = len(items)
        offsets = np.cumsum(offsets)
        self.offsets = offsets
        self.items = np.empty(int(offsets[-1]), dtype=np.int32)
        self.actions = np.empty(int(offsets[-1]), dtype=np.int32)
        for u, items, actions in zip(df["user_index"].values, df["item_seq"].values, df["action_seq"].values):
            s, e = self.offsets[int(u)], self.offsets[int(u) + 1]
            self.items[s:e] = np.asarray(items, dtype=np.int32)
            self.actions[s:e] = np.asarray(actions, dtype=np.int32)

    def slice(self, user_index: int, start: int, end: int):
        s = self.offsets[user_index] + start
        e = self.offsets[user_index] + end
        return self.items[s:e], self.actions[s:e]


class FieldTables:
    """Entity-level value tables used by permutation transforms."""

    def __init__(self, split_df: pd.DataFrame, item_info: pd.DataFrame, n_users: int, n_items: int):
        first = split_df.drop_duplicates("user_index").set_index("user_index").sort_index()
        self.user = np.zeros((n_users, len(DATA_CFG["user_profile_fields"])), dtype=np.int32)
        self.user[first.index.values] = first[DATA_CFG["user_profile_fields"]].values.astype(np.int32)
        self.item = np.zeros((n_items + 1, len(DATA_CFG["target_item_fields"])), dtype=np.int32)
        self.item[item_info["item_index"].values] = item_info[DATA_CFG["target_item_fields"]].values.astype(np.int32)


class KuaiRandData:
    def __init__(self, dataset_key: str):
        self.dataset_key = dataset_key
        root = PATHS["processed_dataset_dirs"][dataset_key]
        self.root = root
        self.splits = {
            "train": pd.read_parquet(root / "train.parquet"),
            "valid": pd.read_parquet(root / "validation.parquet"),
            "test": pd.read_parquet(root / "test.parquet"),
        }
        self.item_info = pd.read_parquet(root / "item_info.parquet")
        with open(root / "vocab.json") as f:
            self.vocab = json.load(f)
        self.history = UserHistoryStore(root / "user_info.parquet")
        self.tables = FieldTables(
            self.splits["train"], self.item_info, self.vocab["n_users"], self.vocab["n_items"]
        )

    def vocab_sizes(self) -> Dict[str, int]:
        vs = {"item_id": self.vocab["n_items"] + 1, "action": self.vocab["num_action_codes"]}
        vs["variable_fields"] = {}
        for f in DATA_CFG["user_profile_fields"]:
            vs["variable_fields"][f] = len(self.vocab["user_vocabs"][f]) + 1
        for f in DATA_CFG["context_fields"]:
            vs["variable_fields"][f] = self.vocab["context_vocabs"][f]
        for f in DATA_CFG["target_item_fields"]:
            vs["variable_fields"][f] = len(self.vocab["item_vocabs"][f]) + 1
        return vs


class QualityTransform:
    """Applies entity-level / row-level permutation maps for exp 1.2."""

    def __init__(self, maps: Dict, tables: FieldTables):
        self.maps = maps  # {"user": {field: arr}, "item": {field: arr}, "context": {split: {field: arr}}}
        self.user_table = tables.user
        self.item_table = tables.item

    def user_value(self, field: str, user_index: int) -> int:
        col = DATA_CFG["user_profile_fields"].index(field)
        return int(self.user_table[self.maps["user"][field][user_index], col])

    def item_value(self, field: str, item_index: int) -> int:
        col = DATA_CFG["target_item_fields"].index(field)
        return int(self.item_table[self.maps["item"][field][item_index], col])

    def context_row(self, field: str, split: str, orig_index: int) -> int:
        return int(self.maps["context"][split][field][orig_index])


class KuaiRandSequenceDataset(Dataset):
    def __init__(self, data: KuaiRandData, split: str, max_len: int,
                 transform: Optional[QualityTransform] = None, min_history: Optional[int] = None,
                 max_rows: Optional[int] = None):
        self.data = data
        self.split = split
        self.max_len = max_len
        self.transform = transform
        df = data.splits[split]
        self.orig_ctx_cols = {f: df[f].values.astype(np.int64) for f in DATA_CFG["context_fields"]}
        if min_history is not None:
            keep = df["position"].values >= min_history
            self.orig_index = np.nonzero(keep)[0]
            df = df.iloc[self.orig_index].reset_index(drop=True)
        else:
            self.orig_index = np.arange(len(df))
        if max_rows is not None and len(df) > max_rows:
            self.orig_index = self.orig_index[:max_rows]
            df = df.iloc[:max_rows].reset_index(drop=True)
        self.df = df
        self.user_index = df["user_index"].values.astype(np.int64)
        self.item_index = df["item_index"].values.astype(np.int64)
        self.position = df["position"].values.astype(np.int64)
        self.label = df["is_click"].values.astype(np.float32)
        self.date = df["date"].values.astype(np.int64)
        self.user_cols = {f: df[f].values.astype(np.int64) for f in DATA_CFG["user_profile_fields"]}
        self.ctx_cols = {f: df[f].values.astype(np.int64) for f in DATA_CFG["context_fields"]}
        self.item_cols = {f: df[f].values.astype(np.int64) for f in DATA_CFG["target_item_fields"]}

    def __len__(self):
        return len(self.df)

    def _field_values(self, i: int):
        out = {}
        if self.transform is None:
            for cols in (self.user_cols, self.ctx_cols, self.item_cols):
                for f, col in cols.items():
                    out[f] = col[i]
        else:
            t = self.transform
            orig = int(self.orig_index[i])
            for f in DATA_CFG["user_profile_fields"]:
                out[f] = t.user_value(f, int(self.user_index[i]))
            for f in DATA_CFG["context_fields"]:
                out[f] = self.orig_ctx_cols[f][t.context_row(f, self.split, orig)]
            for f in DATA_CFG["target_item_fields"]:
                out[f] = t.item_value(f, int(self.item_index[i]))
        return out

    def __getitem__(self, i: int):
        pos = int(self.position[i])
        start = max(0, pos - self.max_len)
        items, actions = self.data.history.slice(int(self.user_index[i]), start, pos)
        sample = {
            "hist_item": torch.from_numpy(items.astype(np.int64)),
            "hist_action": torch.from_numpy(actions.astype(np.int64)),
            "item_id": torch.tensor(self.item_index[i], dtype=torch.int64),
            "label": torch.tensor(self.label[i], dtype=torch.float32),
        }
        sample.update({f: torch.tensor(int(v), dtype=torch.int64)
                       for f, v in self._field_values(i).items()})
        return sample


def collate_batch(samples):
    out = {}
    for key in samples[0]:
        if key == "hist_item":
            out["hist_item"] = torch.nn.utils.rnn.pad_sequence(
                [s["hist_item"] for s in samples], batch_first=True, padding_value=0)
        elif key == "hist_action":
            out["hist_action"] = torch.nn.utils.rnn.pad_sequence(
                [s["hist_action"] for s in samples], batch_first=True, padding_value=0)
        else:
            out[key] = torch.stack([s[key] for s in samples])
    lengths = torch.tensor([len(s["hist_item"]) for s in samples], dtype=torch.long)
    out["hist_mask"] = (torch.arange(out["hist_item"].shape[1]).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return out
