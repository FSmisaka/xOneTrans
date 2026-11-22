"""Static Feature Probe: field information scores Q_j for exp 1.2.

A small model using only the 12 candidate non-sequential fields plus the
target item id is trained on `train_fit` (the earliest 80% of train dates).
Permutation importance Q_j = LogLoss(field permuted) - LogLoss(original) is
measured on `train_calibration` (the remaining 20% of train dates), averaged
over 3 permutations. No validation/test labels are involved.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS
from config.dataset import CONFIG as DATA_CFG
from config.exp_1_2_quality import CONFIG as EXP_CFG
from src.utils.metrics import auc_score, logloss_score
from src.utils.seed import set_seed


class StaticFeatureProbe(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim, hidden_units):
        super().__init__()
        self.fields = list(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes[f], embedding_dim, padding_idx=0) for f in self.fields
        })
        d = embedding_dim * len(self.fields)
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch):
        emb = torch.cat([self.embeddings[f](batch[f]) for f in self.fields], dim=-1)
        return self.mlp(emb).squeeze(-1)


def _row_tensors(df, fields, device):
    return {f: torch.from_numpy(df[f].values.astype(np.int64)).to(device) for f in fields}


@torch.no_grad()
def _logloss(model, tensors, fields, y, idx=None):
    model.eval()
    logits = []
    n = len(y) if idx is None else len(idx)
    bs = 8192
    rows = np.arange(n) if idx is None else idx
    for s in range(0, n, bs):
        r = rows[s:s + bs]
        batch = {f: tensors[f][r] for f in fields}
        logits.append(model(batch).float().cpu().numpy())
    p = 1 / (1 + np.exp(-np.concatenate(logits)))
    return logloss_score(y[rows], p)


def run_static_probe(data, dataset_key: str, device_arg="auto") -> dict:
    cache_fp = PATHS["metadata_root"] / f"static_probe_{dataset_key}.json"
    if cache_fp.exists():
        return json.loads(cache_fp.read_text())

    cfg = EXP_CFG["static_probe"]
    fields = (DATA_CFG["user_profile_fields"] + DATA_CFG["context_fields"]
              + DATA_CFG["target_item_fields"] + ["item_index"])
    set_seed(cfg["seed"])
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    if device_arg != "auto":
        device = torch.device(device_arg)

    train_df = data.splits["train"]
    dates = np.sort(train_df["date"].unique())
    n_fit = int(len(dates) * cfg["train_fit_ratio"])
    fit_df = train_df[train_df["date"] <= dates[n_fit - 1]].reset_index(drop=True)
    cal_df = train_df[train_df["date"] > dates[n_fit - 1]].reset_index(drop=True)

    vocab_sizes = {}
    vs = data.vocab_sizes()
    for f in fields:
        vocab_sizes[f] = vs["variable_fields"].get(f, vs["item_id"])
    model = StaticFeatureProbe(vocab_sizes, cfg["embedding_dim"], cfg["hidden_units"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    y_fit = torch.from_numpy(fit_df["is_click"].values.astype(np.float32)).to(device)
    fit_tensors = _row_tensors(fit_df, fields, device)
    n = len(fit_df)
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg["batch_size"]):
            r = perm[s:s + cfg["batch_size"]]
            batch = {f: fit_tensors[f][r] for f in fields}
            loss = F.binary_cross_entropy_with_logits(model(batch), y_fit[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    cal_tensors = _row_tensors(cal_df, fields, device)
    y_cal = cal_df["is_click"].values.astype(np.float32)
    base = _logloss(model, cal_tensors, fields, y_cal)
    rng = np.random.default_rng(EXP_CFG["importance_seed"])
    importance = {}
    for f in [fl for fl in fields if fl != "item_id"]:
        deltas = []
        for rep in range(EXP_CFG["n_importance_permutations"]):
            perm_idx = rng.permutation(len(cal_df))
            shuffled = {k: v for k, v in cal_tensors.items()}
            shuffled[f] = cal_tensors[f][torch.from_numpy(perm_idx).to(device)]
            deltas.append(_logloss(model, shuffled, fields, y_cal) - base)
        importance[f] = float(np.mean(deltas))
    result = {
        "base_logloss": base,
        "train_fit_rows": int(len(fit_df)),
        "train_calibration_rows": int(len(cal_df)),
        "n_permutations": EXP_CFG["n_importance_permutations"],
        "Q_j": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    cache_fp.write_text(json.dumps(result, indent=2))
    return result
