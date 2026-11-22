"""Per-layer Normalized Effective Rank and sequence-probe diagnostics."""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.sequence_probe import CONFIG as PROBE_CFG
from src.data.dataset import KuaiRandSequenceDataset, collate_batch


def select_diagnostic_indices(dataset: KuaiRandSequenceDataset, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))[: min(n_samples, len(dataset))]
    return np.sort(idx)


def select_ner_token_pairs(lengths: np.ndarray, max_tokens: int, seed: int):
    """Select at most max_tokens (sample-rank, position) pairs deterministically."""
    total = int(lengths.sum())
    rng = np.random.default_rng(seed)
    if total <= max_tokens:
        chosen = np.arange(total)
    else:
        chosen = np.sort(rng.choice(total, size=max_tokens, replace=False))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    rank_arr, pos_arr = [], []
    for r in range(len(lengths)):
        lo, hi = int(offsets[r]), int(offsets[r + 1])
        sel = chosen[(chosen >= lo) & (chosen < hi)] - lo
        rank_arr.append(np.full(len(sel), r))
        pos_arr.append(sel)
    if rank_arr:
        return np.concatenate(rank_arr).astype(np.int64), np.concatenate(pos_arr).astype(np.int64)
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)


@torch.no_grad()
def collect_states(model, dataset: KuaiRandSequenceDataset, sample_indices: np.ndarray,
                   rank_arr: np.ndarray, pos_arr: np.ndarray, batch_size: int,
                   device: torch.device, pool_last_layer: bool):
    """Per-layer selected token states (FP32, CPU) and optional masked-mean
    pooled last-layer vectors with labels."""
    model.eval()
    loader = DataLoader(Subset(dataset, sample_indices.tolist()), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_batch)
    token_chunks: List[List[torch.Tensor]] = None
    pooled, labels = [], []
    consumed = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, return_sequence_states=True)
        states = out["states"]
        b = states[0].shape[0]
        if token_chunks is None:
            token_chunks = [[] for _ in states]
        sel = (rank_arr >= consumed) & (rank_arr < consumed + b)
        ranks_l = rank_arr[sel] - consumed
        poss_l = pos_arr[sel]
        for li, st in enumerate(states):
            st = st.float()
            flat = st.view(-1, st.shape[-1])[ranks_l * st.shape[1] + poss_l]
            if flat.shape[0]:
                token_chunks[li].append(flat.cpu())
        if pool_last_layer:
            mask = out["hist_mask"]
            last = states[-1].float()
            pooled.append(((last * mask.unsqueeze(-1)).sum(1)
                           / mask.sum(1, keepdim=True).clamp_min(1)).cpu())
            labels.append(batch["label"].cpu())
        consumed += b
    tokens = [torch.cat(ch, dim=0) if ch else torch.zeros(0, model.token_dim) for ch in token_chunks]
    result = {"tokens": tokens}
    if pool_last_layer:
        result["pooled"] = torch.cat(pooled)
        result["labels"] = torch.cat(labels)
    return result


def normalized_effective_rank(H: torch.Tensor) -> float:
    H = H.to(torch.float32)
    H = H - H.mean(dim=0, keepdim=True)
    sigma = torch.linalg.svdvals(H)
    p = sigma / sigma.sum()
    er = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()).item()
    return er / H.shape[1]


def compute_layer_ner(tokens: List[torch.Tensor]) -> Dict[str, float]:
    return {
        ("backbone_input" if li == 0 else f"layer_{li}"): normalized_effective_rank(H)
        for li, H in enumerate(tokens)
    }


def fit_sequence_probe(train_X: np.ndarray, train_y: np.ndarray,
                       valid_X, valid_y, test_X, test_y) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    lr = LogisticRegression(C=PROBE_CFG["probe_C"], max_iter=PROBE_CFG["probe_max_iter"],
                            solver=PROBE_CFG["probe_solver"])
    lr.fit(train_X, train_y)
    return {
        "probe_valid_auc": float(roc_auc_score(valid_y, lr.predict_proba(valid_X)[:, 1])),
        "probe_test_auc": float(roc_auc_score(test_y, lr.predict_proba(test_X)[:, 1])),
    }
