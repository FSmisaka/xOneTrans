"""Exp4: continuous spectral OneTrans mixture (SPEC4).

One frozen OneTrans prefix, one shared last block and one shared CTR tower.
A fixed train-reference spectral transform creates K alpha links; fusion is
either prediction-level expert mixing or contextual representation mixing.
No NER loss is used in training.
"""

import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.exp_4_continuous_spectral import CONFIG as EXP_4
from config.onetrans import CONFIG as MODEL_CFG
from config.paths import CONFIG as PATHS
from config.sequence_probe import CONFIG as PROBE_CFG
from src.data.dataset import KuaiRandSequenceDataset, collate_batch
from src.experiments import figures
from src.experiments.diagnostics import (normalized_effective_rank,
                                         select_diagnostic_indices,
                                         select_ner_token_pairs)
from src.experiments.runner import (accum_steps_for, dist_rank, ensure_dist, get_data,
                                    peak_memory_mb, resolve_device)
from src.models.continuous_spectral_onetrans import ContinuousSpectralOneTrans
from src.models.spectral_transform import (apply_spectral_transform,
                                           build_spectral_transform,
                                           fit_spectral_basis)
from src.utils.logging import setup_logger
from src.utils.metrics import auc_score, logloss_score
from src.utils.seed import set_seed

EXP_ID = EXP_4["experiment"]
EXP4_ALPHAS = [float(a) for a in EXP_4["spectral"]["alpha_nodes"]]


def _canonical_alphas(values) -> List[float]:
    return [min(EXP4_ALPHAS, key=lambda x: abs(x - float(v))) for v in values]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def _write_yaml(path: Path, payload) -> None:
    _write_json(path, payload)


def _sha256(payload) -> str:
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True).encode()).hexdigest()


def _checkpoint_stat(path: Path):
    if not path.exists():
        return None
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _validate_spectral_config() -> None:
    alphas = [float(a) for a in EXP_4["spectral"]["alpha_nodes"]]
    weights = [float(q) for q in EXP_4["spectral"]["quadrature_weights"]]
    if len(alphas) != len(weights) or not alphas:
        raise ValueError("alpha_nodes and quadrature_weights must have the same nonzero length")
    if any(a <= 0 for a in alphas):
        raise ValueError("all alpha nodes must be positive")
    if any(q < 0 for q in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("quadrature weights must be non-negative and sum to 1")
    if EXP_4["spectral"].get("bracket_alpha_one", True):
        if alphas[0] >= 1.0 or alphas[-1] <= 1.0:
            raise ValueError("alpha nodes must bracket alpha=1")


def _baseline_path(seed: int) -> Path:
    return Path(EXP_4["baseline_checkpoint"].format(seed=seed))


def _run_signature(seed: int, condition: str, spec: Dict, dataset: str, baseline: Path,
                   debug: bool) -> str:
    payload = {
        "experiment": EXP_ID, "condition": condition, "condition_spec": spec,
        "seed": seed, "dataset": dataset, "debug": debug,
        "debug_cfg": EXP_4["debug"] if debug else None,
        "model_cfg": MODEL_CFG, "calibration": EXP_4["calibration"],
        "spectral": EXP_4["spectral"], "training": EXP_4["training"],
        "baseline_checkpoint": str(baseline),
        "baseline_checkpoint_stat": _checkpoint_stat(baseline),
    }
    return _sha256(payload)


def resolve_conditions(names: Optional[List[str]] = None) -> List:
    selected = names or EXP_4["default_conditions"]
    missing = [n for n in selected if n not in EXP_4["conditions"]]
    if missing:
        raise ValueError(f"unknown conditions: {missing}")
    return [(n, dict(EXP_4["conditions"][n])) for n in selected]


class CsvLogger:
    def __init__(self, path: Path, fields: List[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row: Dict) -> None:
        self.writer.writerow({k: row.get(k, "") for k in self.fields})

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


EPOCH_FIELDS = ["epoch", "step", "seed", "condition", "loss_mix", "loss_nodes",
                "loss_total", "node_aux_loss_weight", "grad_norm",
                "valid_auc", "valid_logloss", "valid_auc_mix", "valid_logloss_mix",
                "valid_auc_weighted_logit", "valid_logloss_weighted_logit"]
STEP_FIELDS = ["epoch", "step", "micro_step", "seed", "condition", "loss_mix",
               "loss_nodes", "loss_total", "node_aux_loss_weight", "grad_norm_total"]


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {"auc": auc_score(y_true, y_pred),
            "logloss": logloss_score(y_true, y_pred),
            "n": int(len(y_true))}


@torch.no_grad()
def evaluate_model(model, loader, device, compute_nodes: bool = True) -> Dict:
    core = model.module if hasattr(model, "module") else model
    core.eval()
    y_true, primary, weights = [], [], []
    node_logits, node_probs = [], []
    alpha_nodes = None
    t0 = time.time()
    n_samples = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = core(batch)
        y_true.append(batch["label"].cpu().numpy())
        primary.append(out["pred"].float().cpu().numpy())
        weights.append(out["weights"].float().cpu().numpy())
        if alpha_nodes is None:
            alpha_nodes = _canonical_alphas(out["alpha_nodes"].detach().cpu().tolist())
        if compute_nodes and "logits_by_alpha" in out:
            logits = out["logits_by_alpha"].float()
            node_logits.append(logits.cpu().numpy())
            node_probs.append(torch.sigmoid(logits).cpu().numpy())
        n_samples += int(batch["label"].numel())
    y = np.concatenate(y_true)
    pred = np.concatenate(primary)
    result = {"y_true": y, "primary": pred, "primary_metrics": _metric_dict(y, pred),
              "weights": np.concatenate(weights), "alpha_nodes": alpha_nodes,
              "elapsed": time.time() - t0, "n": n_samples, "kind": getattr(core, "kind", None)}
    if node_logits:
        logits = np.concatenate(node_logits)
        probs = np.concatenate(node_probs)
        result["node_logits"] = logits
        result["node_probs"] = probs
        result["node_metrics"] = {
            str(a): _metric_dict(y, probs[:, k]) for k, a in enumerate(alpha_nodes)}
        if getattr(core, "kind", None) == "expert_mix":
            w = result["weights"]
            wl = (w * logits).sum(axis=1)
            wl_prob = 1.0 / (1.0 + np.exp(-wl))
            result["weighted_logit_metrics"] = _metric_dict(y, wl_prob)
    return result


@torch.no_grad()
def collect_calibration_tokens(base, dataset: KuaiRandSequenceDataset, device,
                               max_tokens: int, seed: int, batch_size: int) -> Dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(dataset))
    loader = DataLoader(Subset(dataset, perm.tolist()), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_batch,
                        num_workers=MODEL_CFG["num_workers"])
    base.eval()
    chunks, used_indices = [], []
    total = 0
    offset = 0
    for batch in loader:
        bs = int(batch["label"].shape[0])
        batch = {k: v.to(device) for k, v in batch.items()}
        s_pre, _, mask = base.encode_prefix(batch)
        valid = s_pre[mask.bool()].float().cpu()
        if valid.shape[0] > 0 and total < max_tokens:
            take = min(max_tokens - total, valid.shape[0])
            chunks.append(valid[:take])
            counts = mask.sum(dim=1).long().cpu()
            cumulative = torch.cumsum(counts, dim=0)
            touched = bs if take >= valid.shape[0] else int((cumulative < take).sum()) + 1
            used_indices.extend(perm[offset:offset + touched].tolist())
            total += take
        offset += bs
        if total >= max_tokens:
            break
    if not chunks:
        raise RuntimeError("calibration collected no valid S tokens")
    tokens = torch.cat(chunks, dim=0)
    if tokens.shape[0] < 2:
        raise RuntimeError(f"calibration has only {tokens.shape[0]} tokens")
    return {"tokens": tokens, "sample_indices": used_indices,
            "n_tokens": int(tokens.shape[0]), "n_samples": len(used_indices)}


def build_basis_payload(calibration: Dict, dataset_key: str, seed: int,
                        baseline_path: Path, run_signature: str, debug: bool) -> Dict:
    if EXP_4["calibration"]["split"] != "train":
        raise ValueError("Exp4 calibration must use the train split")
    metadata = {
        "calibration_split": "train",
        "dataset": dataset_key,
        "seed": seed,
        "baseline_checkpoint": str(baseline_path),
        "run_signature": run_signature,
        "debug": debug,
        "sample_indices": calibration["sample_indices"],
        "n_tokens": calibration["n_tokens"],
        "n_samples": calibration["n_samples"],
        "calibration_config": EXP_4["calibration"],
        "spectral_config": EXP_4["spectral"],
    }
    basis = fit_spectral_basis(
        calibration["tokens"], rank_relative_tolerance=EXP_4["calibration"]["rank_relative_tolerance"],
        svd_dtype=EXP_4["calibration"]["svd_dtype"], metadata=metadata)
    alphas = [float(a) for a in EXP_4["spectral"]["alpha_nodes"]]
    transforms = []
    for alpha in alphas:
        transform = build_spectral_transform(
            basis["sigma"], basis["V"], alpha, token_dim=basis["sigma"].numel(),
            rank_relative_tolerance=EXP_4["calibration"]["rank_relative_tolerance"])
        x_ref = apply_spectral_transform(calibration["tokens"], None, basis["mu_ref"],
                                         transform["T"], alpha=alpha)
        transform["calibration_ner"] = float(normalized_effective_rank(x_ref))
        transforms.append(transform)
    return {
        "basis": basis,
        "transforms": transforms,
        "alpha_nodes": alphas,
        "quadrature_weights": [float(q) for q in EXP_4["spectral"]["quadrature_weights"]],
        "reference_ner": [float(tr["analytic_ner"]) for tr in transforms],
        "calibration_ner": [float(tr["calibration_ner"]) for tr in transforms],
        "metadata": basis["metadata"],
    }


def _ensure_basis(model, datasets, device, run_dir: Path, signature: str,
                  dataset_key: str, seed: int, baseline_path: Path, debug: bool,
                  rank: int, ddp: bool) -> Dict:
    basis_path = run_dir / "spectral_basis.pt"
    if ddp:
        payload = None
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            if basis_path.exists():
                try:
                    cached = torch.load(basis_path, map_location="cpu")
                except Exception:
                    cached = None
                if cached and cached.get("metadata", {}).get("run_signature") == signature:
                    payload = cached
            if payload is None:
                result = collect_calibration_tokens(
                    model.base, datasets["train"], device,
                    EXP_4["calibration"]["max_tokens"] if not debug
                    else EXP_4["debug"]["calibration_max_tokens"],
                    EXP_4["calibration"]["seed"], EXP_4["calibration"]["batch_size"])
                payload = build_basis_payload(result, dataset_key, seed, baseline_path,
                                              signature, debug)
                tmp_path = run_dir / "spectral_basis.tmp"
                torch.save(payload, tmp_path)
                os.replace(tmp_path, basis_path)
        # A barrier only synchronizes processes; it does not guarantee that a
        # file written on a network filesystem is visible to other ranks.
        # Broadcast the payload in memory so every rank uses the exact same basis.
        objects = [payload]
        torch.distributed.broadcast_object_list(objects, src=0, device=device)
        payload = objects[0]
        if payload is None:
            raise RuntimeError("rank 0 failed to build the spectral basis")
        return payload
    run_dir.mkdir(parents=True, exist_ok=True)
    if basis_path.exists():
        try:
            cached = torch.load(basis_path, map_location="cpu")
        except Exception:
            cached = None
        if cached and cached.get("metadata", {}).get("run_signature") == signature:
            return cached
    result = collect_calibration_tokens(
        model.base, datasets["train"], device,
        EXP_4["calibration"]["max_tokens"] if not debug
        else EXP_4["debug"]["calibration_max_tokens"],
        EXP_4["calibration"]["seed"], EXP_4["calibration"]["batch_size"])
    payload = build_basis_payload(result, dataset_key, seed, baseline_path,
                                  signature, debug)
    tmp_path = run_dir / "spectral_basis.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, basis_path)
    return payload


def _diagnostic_setup(dataset: KuaiRandSequenceDataset, n_samples: int, max_tokens: int,
                      max_len: int):
    idx = select_diagnostic_indices(dataset, n_samples, PROBE_CFG["diagnostic_seed"])
    lengths = np.minimum(dataset.position[idx], max_len)
    rank_arr, pos_arr = select_ner_token_pairs(lengths, max_tokens, PROBE_CFG["ner_seed"])
    return idx, rank_arr, pos_arr


@torch.no_grad()
def collect_ner_tokens(model, dataset, diag, batch_size: int, device) -> Dict:
    idx, rank_arr, pos_arr = diag
    core = model.module if hasattr(model, "module") else model
    core.eval()
    loader = DataLoader(Subset(dataset, idx.tolist()), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_batch)
    chunks_pre = []
    chunks_alpha = None
    chunks_final = None
    chunks_mix = []
    chunks_mix_final = []
    alpha_nodes = None
    consumed = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = core(batch, return_states=True)
        b = out["s_pre"].shape[0]
        if alpha_nodes is None:
            alpha_nodes = _canonical_alphas(out["alpha_nodes"].detach().cpu().tolist())
        sel = (rank_arr >= consumed) & (rank_arr < consumed + b)
        ranks_l = rank_arr[sel] - consumed
        poss_l = pos_arr[sel]
        pre = out["s_pre"].float()
        flat = pre.view(-1, pre.shape[-1])[ranks_l * pre.shape[1] + poss_l]
        if flat.shape[0]:
            chunks_pre.append(flat.cpu())
        s_alpha = out.get("s_alpha")
        if s_alpha is not None:
            s_alpha = s_alpha.float()
            if chunks_alpha is None:
                chunks_alpha = [[] for _ in range(s_alpha.shape[1])]
                chunks_final = [[] for _ in range(s_alpha.shape[1])]
            for k in range(s_alpha.shape[1]):
                state = s_alpha[:, k]
                flat = state.reshape(-1, state.shape[-1])[ranks_l * state.shape[1] + poss_l]
                if flat.shape[0]:
                    chunks_alpha[k].append(flat.cpu())
                if "s_final" in out:
                    state_final = out["s_final"][:, k].float()
                    flat_final = state_final.reshape(
                        -1, state_final.shape[-1])[ranks_l * state_final.shape[1] + poss_l]
                    if flat_final.shape[0]:
                        chunks_final[k].append(flat_final.cpu())
        if "s_mix" in out:
            s_mix = out["s_mix"].float()
            flat_mix = s_mix.reshape(-1, s_mix.shape[-1])[ranks_l * s_mix.shape[1] + poss_l]
            if flat_mix.shape[0]:
                chunks_mix.append(flat_mix.cpu())
            s_mix_final = out["s_final_mix"].float()
            flat_mix_final = s_mix_final.reshape(
                -1, s_mix_final.shape[-1])[ranks_l * s_mix_final.shape[1] + poss_l]
            if flat_mix_final.shape[0]:
                chunks_mix_final.append(flat_mix_final.cpu())
        consumed += b
    token_dim = core.token_dim
    pre_tokens = torch.cat(chunks_pre, 0) if chunks_pre else torch.zeros(0, token_dim)
    alpha_tokens = [torch.cat(ch, 0) if ch else torch.zeros(0, token_dim)
                    for ch in (chunks_alpha or [])]
    final_tokens = [torch.cat(ch, 0) if ch else torch.zeros(0, token_dim)
                    for ch in (chunks_final or [])]
    mix_tokens = torch.cat(chunks_mix, 0) if chunks_mix else None
    mix_final_tokens = torch.cat(chunks_mix_final, 0) if chunks_mix_final else None
    return {"alpha_nodes": alpha_nodes, "s_pre": pre_tokens,
            "s_alpha": alpha_tokens, "s_final": final_tokens,
            "s_mix": mix_tokens, "s_mix_final": mix_final_tokens}


def _ner(tokens: Optional[torch.Tensor]):
    if tokens is None or tokens.shape[0] < 2:
        return None
    return float(normalized_effective_rank(tokens))


def compute_ner_metrics(model, datasets, device, max_len: int, debug: bool) -> Dict:
    eval_cfg = EXP_4["evaluation"]
    n_samples = (EXP_4["debug"]["diagnostic_samples"] if debug
                 else eval_cfg["diagnostic_samples"])
    max_tokens = (EXP_4["debug"]["ner_max_tokens"] if debug else eval_cfg["ner_max_tokens"])
    ner_bs = (EXP_4["debug"]["ner_batch_size"] if debug else eval_cfg["ner_batch_size"])
    valid_diag = _diagnostic_setup(datasets["valid"], n_samples, max_tokens, max_len)
    test_diag = _diagnostic_setup(datasets["test"], n_samples, max_tokens, max_len)
    valid = collect_ner_tokens(model, datasets["valid"], valid_diag, ner_bs, device)
    test = collect_ner_tokens(model, datasets["test"], test_diag, ner_bs, device)
    alpha_nodes = valid["alpha_nodes"]
    result = {
        "alpha_nodes": alpha_nodes,
        "s_pre": {"valid": _ner(valid["s_pre"]), "test": _ner(test["s_pre"])},
        "pre_final": {
            "valid": [_ner(t) for t in valid["s_alpha"]],
            "test": [_ner(t) for t in test["s_alpha"]],
        },
        "post_final": {
            "valid": [_ner(t) for t in valid["s_final"]],
            "test": [_ner(t) for t in test["s_final"]],
        },
        "representation_mix": None,
    }
    if valid["s_mix"] is not None:
        result["representation_mix"] = {
            "valid_pre": _ner(valid["s_mix"]), "valid_post": _ner(valid["s_mix_final"]),
            "test_pre": _ner(test["s_mix"]), "test_post": _ner(test["s_mix_final"]),
        }
    return result


def _quantiles(values: np.ndarray):
    if len(values) == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"),
                "p90": float("nan")}
    return {"mean": float(values.mean()), "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)), "p90": float(np.percentile(values, 90))}


def compute_weight_stats(weights: np.ndarray, alpha_nodes: List[float],
                         lengths: np.ndarray, groups: np.ndarray,
                         group_field: str, buckets: List[int]) -> Dict:
    weights = np.asarray(weights, dtype=float)
    alpha = np.asarray(alpha_nodes, dtype=float)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    expected_alpha = weights @ alpha
    overall = {
        "n_samples": int(weights.shape[0]),
        "mean": weights.mean(axis=0).tolist(),
        "std": weights.std(axis=0).tolist(),
        "min": weights.min(axis=0).tolist(),
        "max": weights.max(axis=0).tolist(),
        "p10": np.percentile(weights, 10, axis=0).tolist(),
        "p50": np.percentile(weights, 50, axis=0).tolist(),
        "p90": np.percentile(weights, 90, axis=0).tolist(),
        "entropy": _quantiles(entropy),
        "expected_alpha": _quantiles(expected_alpha),
    }
    by_history_len = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        if mask.any():
            by_history_len.append({"bucket": f"[{lo},{hi})", "n": int(mask.sum()),
                                   "mean_weights": weights[mask].mean(axis=0).tolist()})
    by_group = []
    for value in np.unique(groups):
        mask = groups == value
        if mask.any():
            by_group.append({"group": int(value), "n": int(mask.sum()),
                             "mean_weights": weights[mask].mean(axis=0).tolist()})
    return {"group_field": group_field, "overall": overall,
            "by_history_len": by_history_len, "by_group": by_group}


def _weight_stats_for_split(result: Dict, dataset: KuaiRandSequenceDataset,
                            max_len: int) -> Dict:
    lengths = np.minimum(dataset.position, max_len)
    field = EXP_4["evaluation"]["group_field"]
    groups = (dataset.item_cols[field] if field in dataset.item_cols
              else np.zeros(len(dataset), dtype=np.int64))
    return compute_weight_stats(result["weights"], result["alpha_nodes"], lengths, groups,
                                field, EXP_4["evaluation"]["history_length_buckets"])


def _node_rows(node_metrics: Dict) -> List[Dict]:
    rows = []
    for alpha, metrics in (node_metrics or {}).items():
        rows.append({"alpha": alpha, "auc": metrics["auc"],
                     "logloss": metrics["logloss"], "n": metrics["n"]})
    return rows


def _setup_data(dataset_key: str, debug: bool) -> Dict:
    data = get_data(dataset_key)
    max_len = MODEL_CFG["max_history_len"]
    caps = EXP_4["debug"] if debug else {}
    datasets = {
        split: KuaiRandSequenceDataset(data, split, max_len,
                                       max_rows=caps.get(f"{split}_samples"))
        for split in ("train", "valid", "test")
    }
    return {"data": data, "datasets": datasets, "max_len": max_len}


def _load_baseline(model, baseline_path: Path) -> None:
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Exp3 single_baseline checkpoint not found: {baseline_path}. "
            "Run Exp3 single_baseline for this seed first.")
    ckpt = torch.load(baseline_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
            break
    model.base.load_state_dict(state, strict=True)
    model.baseline_checkpoint = str(baseline_path)


def _train_condition(model, datasets, seed: int, condition: str, spec: Dict,
                     signature: str, run_dir: Path, device, ddp: bool,
                     logger, debug: bool, max_len: int) -> Dict:
    world_size = torch.distributed.get_world_size() if ddp else 1
    accum_multiplier = int(EXP_4["training"]["accum_steps_multiplier"])
    accum_steps = max(1, accum_steps_for(MODEL_CFG["max_history_len"],
                                         MODEL_CFG["batch_size"], device)
                      * accum_multiplier)
    micro_batch = max(1, MODEL_CFG["batch_size"] // (accum_steps * world_size))
    eval_bs = max(256, MODEL_CFG["eval_batch_size"] * MODEL_CFG["max_history_len"]
                  // max_len)
    train_sampler = (torch.utils.data.DistributedSampler(datasets["train"], shuffle=True,
                                                         seed=seed) if ddp else None)
    shuffle_generator = None
    if train_sampler is None:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=micro_batch,
                              shuffle=train_sampler is None, sampler=train_sampler,
                              generator=shuffle_generator, collate_fn=collate_batch,
                              num_workers=MODEL_CFG["num_workers"], drop_last=False)
    valid_loader = DataLoader(datasets["valid"], batch_size=eval_bs, shuffle=False,
                              collate_fn=collate_batch, num_workers=MODEL_CFG["num_workers"])
    test_loader = DataLoader(datasets["test"], batch_size=eval_bs, shuffle=False,
                             collate_fn=collate_batch, num_workers=MODEL_CFG["num_workers"])
    optimizer = torch.optim.Adam(
        (model.module if ddp else model).param_groups(
            MODEL_CFG["dense_learning_rate"], MODEL_CFG["sparse_learning_rate"]),
        weight_decay=MODEL_CFG["weight_decay"])
    params = [p for p in (model.module if ddp else model).parameters() if p.requires_grad]
    amp_dtype = {"none": None, "bf16": torch.bfloat16}.get(MODEL_CFG["amp_dtype"])
    n_epochs = EXP_4["debug"]["epochs"] if debug else MODEL_CFG["epochs"]
    aux_weight = float(spec.get("node_aux_loss_weight",
                                EXP_4["training"]["node_aux_loss_weight"]
                                if spec["kind"] == "expert_mix" else 0.0))
    kind = spec["kind"]
    epoch_logger = CsvLogger(run_dir / "metrics_by_epoch.csv", EPOCH_FIELDS) if not ddp or dist_rank() == 0 else None
    step_logger = CsvLogger(run_dir / "loss_curves.csv", STEP_FIELDS) if not ddp or dist_rank() == 0 else None
    best_auc, best_epoch, best_state = -1.0, -1, None
    patience_left = MODEL_CFG["early_stop_patience"]
    global_step = 0
    micro_step = 0
    epochs_run = 0
    train_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        sums = defaultdict(float)
        n_micro = 0
        grad_norms = []
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            is_update = ((step + 1) % accum_steps == 0) or (step + 1 == len(train_loader))
            sync_ctx = (model.no_sync() if (ddp and not is_update)
                        else contextlib.nullcontext())
            with sync_ctx:
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        out = model(batch)
                else:
                    out = model(batch)
                label = batch["label"]
                if kind == "expert_mix":
                    mix_prob = out["mix_prob"].float()
                    loss_mix = F.binary_cross_entropy(mix_prob.clamp(1e-7, 1.0 - 1e-7), label)
                    if aux_weight > 0:
                        logits = out["logits_by_alpha"].float()
                        q = out["quad_weights"].float()
                        node_losses = torch.stack([
                            F.binary_cross_entropy_with_logits(logits[:, k], label)
                            for k in range(logits.shape[1])])
                        loss_nodes = (q * node_losses).sum()
                    else:
                        loss_nodes = torch.zeros((), device=device)
                    loss = loss_mix + aux_weight * loss_nodes
                else:
                    loss_mix = F.binary_cross_entropy_with_logits(out["logit"].float(), label)
                    loss_nodes = torch.zeros((), device=device)
                    loss = loss_mix
                (loss / accum_steps).backward()
            if is_update:
                grad_norm = _grad_norm(params)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_norms.append(grad_norm)
                global_step += 1
            else:
                grad_norm = ""
            row = {
                "epoch": epoch + 1, "step": global_step, "micro_step": micro_step,
                "seed": seed, "condition": condition,
                "loss_mix": float(loss_mix.detach()),
                "loss_nodes": float(loss_nodes.detach()) if kind == "expert_mix" else "",
                "loss_total": float(loss.detach()), "node_aux_loss_weight": aux_weight,
                "grad_norm_total": grad_norm,
            }
            for key in ("loss_mix", "loss_nodes", "loss_total"):
                if row[key] != "":
                    sums[key] += row[key]
            if step_logger is not None:
                step_logger.write(row)
            n_micro += 1
            micro_step += 1
        epochs_run = epoch + 1
        mean_losses = {k: v / n_micro for k, v in sums.items()}
        if dist_rank() == 0:
            valid = evaluate_model(model, valid_loader, device,
                                   compute_nodes=EXP_4["evaluation"]["compute_node_metrics"])
            primary = valid["primary_metrics"]
            epoch_row = {
                "epoch": epoch + 1, "step": global_step, "seed": seed,
                "condition": condition,
                "loss_mix": mean_losses.get("loss_mix"),
                "loss_nodes": mean_losses.get("loss_nodes", ""),
                "loss_total": mean_losses.get("loss_total"),
                "node_aux_loss_weight": aux_weight,
                "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "valid_auc": primary["auc"], "valid_logloss": primary["logloss"],
                "valid_auc_mix": primary["auc"], "valid_logloss_mix": primary["logloss"],
                "valid_auc_weighted_logit": (valid.get("weighted_logit_metrics") or {}).get("auc", ""),
                "valid_logloss_weighted_logit": (valid.get("weighted_logit_metrics") or {}).get("logloss", ""),
            }
            epoch_logger.write(epoch_row)
            epoch_logger.flush()
            logger.info(f"epoch {epoch + 1}: valid_auc={primary['auc']:.6f} "
                        f"loss={float(loss.detach()):.6f}")
            auc_primary = primary["auc"]
        else:
            auc_primary = None
        stop = torch.zeros(1, device=device)
        if auc_primary is not None:
            if auc_primary > best_auc:
                best_auc, best_epoch = auc_primary, epoch + 1
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()})
                patience_left = MODEL_CFG["early_stop_patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    stop.fill_(1.0)
        if ddp:
            torch.distributed.broadcast(stop, src=0)
        if stop.item() > 0:
            if logger:
                logger.info(f"early stop at epoch {epoch + 1} (best epoch {best_epoch})")
            break
    train_seconds = time.time() - train_start
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if epoch_logger is not None:
        epoch_logger.close()
        step_logger.close()
    if dist_rank() != 0:
        return {"valid": None, "test": None, "train_seconds": train_seconds,
                "epochs_run": epochs_run, "best_epoch": best_epoch,
                "best_valid_auc": best_auc}
    valid = evaluate_model(model, valid_loader, device,
                           compute_nodes=EXP_4["evaluation"]["compute_node_metrics"])
    test = evaluate_model(model, test_loader, device,
                          compute_nodes=EXP_4["evaluation"]["compute_node_metrics"])
    return {"valid": valid, "test": test, "train_seconds": train_seconds,
            "epochs_run": epochs_run, "best_epoch": best_epoch,
            "best_valid_auc": best_auc, "micro_batch": micro_batch,
            "accum_steps": accum_steps}


def _split_ner_payload(ner: Dict, split: str) -> Dict:
    rm = ner.get("representation_mix")
    representation = None
    if rm:
        representation = {"pre": rm.get(f"{split}_pre"), "post": rm.get(f"{split}_post")}
    return {"alpha_nodes": ner["alpha_nodes"], "s_pre": ner["s_pre"][split],
            "pre_final": ner["pre_final"][split], "post_final": ner["post_final"][split],
            "representation_mix": representation}


def _combine_split_payloads(valid_payload: Dict, test_payload: Dict) -> Dict:
    valid_ner = valid_payload["ner"]
    test_ner = test_payload["ner"]
    rm = {"valid_pre": (valid_ner.get("representation_mix") or {}).get("pre"),
          "valid_post": (valid_ner.get("representation_mix") or {}).get("post"),
          "test_pre": (test_ner.get("representation_mix") or {}).get("pre"),
          "test_post": (test_ner.get("representation_mix") or {}).get("post")}
    if all(v is None for v in rm.values()):
        rm = None
    valid = valid_payload
    test = test_payload
    per_seed = {
        "experiment": valid["experiment"], "condition": valid["condition"],
        "dataset": valid["dataset"], "seed": valid["seed"], "kind": valid["kind"],
        "fusion": valid.get("fusion"), "alpha": valid.get("alpha"),
        "node_aux_loss_weight": valid.get("node_aux_loss_weight", 0.0),
        "variable": {"kind": valid["kind"], "fusion": valid.get("fusion"),
                     "alpha": valid.get("alpha"),
                     "node_aux_loss_weight": valid.get("node_aux_loss_weight", 0.0)},
        "valid_primary": valid["primary"], "test_primary": test["primary"],
        "node_metrics": {"valid": valid.get("nodes"), "test": test.get("nodes")},
        "weighted_logit": {"valid": valid.get("weighted_logit"),
                           "test": test.get("weighted_logit")},
        "ner": {"alpha_nodes": valid["ner"]["alpha_nodes"],
                "s_pre": {"valid": valid["ner"]["s_pre"], "test": test["ner"]["s_pre"]},
                "pre_final": {"valid": valid["ner"]["pre_final"],
                              "test": test["ner"]["pre_final"]},
                "post_final": {"valid": valid["ner"]["post_final"],
                               "test": test["ner"]["post_final"]},
                "representation_mix": rm},
        "weight_stats": {"valid": valid.get("weight_stats"), "test": test.get("weight_stats")},
        "reference_ner": valid.get("reference_ner"),
        "calibration_ner": valid.get("calibration_ner"),
        "spectral_alpha_nodes": valid.get("spectral_alpha_nodes"),
        "spectral_basis_metadata": valid.get("spectral_basis"),
        "model_summary": valid.get("model_summary"),
        "training": valid.get("training"),
        "checkpoint": valid.get("checkpoint"),
        "run_signature": valid.get("run_signature"),
    }
    return per_seed


def run_condition_4(condition: str, spec: Dict, dataset_key: str, seed: int,
                    device_arg: str = "auto", debug: bool = False,
                    skip_existing: bool = False) -> Optional[Dict]:
    ensure_dist()
    ddp = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    rank = torch.distributed.get_rank() if ddp else 0
    baseline_path = _baseline_path(seed)
    signature = _run_signature(seed, condition, spec, dataset_key, baseline_path, debug)
    run_dir = PATHS["results_root"] / "runs" / EXP_ID / condition / f"seed_{seed}"
    metrics_fp = run_dir / "metrics_test.json"
    if ddp:
        device = (torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
                  if torch.cuda.is_available() else torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        device = resolve_device(device_arg)

    skipped_payload = None
    skip = False
    if skip_existing and rank == 0 and metrics_fp.exists():
        try:
            payload = json.loads(metrics_fp.read_text())
        except Exception:
            payload = None
        if payload is not None and payload.get("run_signature") == signature:
            skip = True
            skipped_payload = payload
            print(f"[skip] {metrics_fp} already exists with matching signature")
            per_seed = PATHS["metrics_root"] / EXP_ID / f"{condition}__{seed}.json"
            if not per_seed.exists():
                combined = run_dir / "metrics_combined.json"
                if combined.exists():
                    _write_json(per_seed, json.loads(combined.read_text()))
                else:
                    valid_payload = json.loads((run_dir / "metrics_valid.json").read_text())
                    test_payload = json.loads((run_dir / "metrics_test.json").read_text())
                    _write_json(per_seed, _combine_split_payloads(valid_payload, test_payload))
    if ddp:
        # All ranks must make the same skip decision even if the shared file is
        # not visible to every rank at the same instant.
        decision = [skip]
        torch.distributed.broadcast_object_list(decision, src=0, device=device)
        skip = bool(decision[0])
    if skip:
        return skipped_payload if rank == 0 else None

    logger = (setup_logger(f"{EXP_ID}_{condition}_{seed}",
                           PATHS["logs_root"] / EXP_ID / condition / f"{seed}.log")
              if rank == 0 else None)
    if logger:
        logger.info(f"experiment={EXP_ID} condition={condition} dataset={dataset_key} "
                    f"seed={seed} device={device} ddp={ddp}")
    _validate_spectral_config()
    if int(EXP_4["model"]["trainable_last_n_layers"]) != 1:
        raise ValueError("Exp4 phase 1 supports trainable_last_n_layers=1 only")
    if not EXP_4["training"]["freeze_prefix"]:
        raise ValueError("Exp4 phase 1 requires freeze_prefix=True")
    if MODEL_CFG["reduction_ratio"] != 0.0:
        raise ValueError("Exp4 requires reduction_ratio=0.0 for stable token positions")

    set_seed(seed)
    bundle = _setup_data(dataset_key, debug)
    data, datasets, max_len = bundle["data"], bundle["datasets"], bundle["max_len"]
    if logger:
        logger.info("samples: " + ", ".join(f"{s}={len(d)}" for s, d in datasets.items()))
    model_cfg = {**MODEL_CFG, "use_residual_gate": False}
    model = ContinuousSpectralOneTrans(
        data.vocab_sizes(), model_cfg, spec, EXP_4["spectral"],
        gate_hidden_dim=EXP_4["model"]["gate_hidden_dim"],
        gate_dropout=EXP_4["model"]["gate_dropout"])
    _load_baseline(model, baseline_path)
    model.freeze_prefix()
    model = model.to(device)
    if logger:
        logger.info(f"baseline checkpoint loaded: {baseline_path}")
    basis_payload = _ensure_basis(model, datasets, device, run_dir, signature,
                                  dataset_key, seed, baseline_path, debug, rank, ddp)
    model.set_spectral_basis(basis_payload["basis"], basis_payload["transforms"],
                             basis_payload["alpha_nodes"], basis_payload["quadrature_weights"])
    if ddp:
        model = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
                 if torch.cuda.is_available() else torch.nn.parallel.DistributedDataParallel(model))
    if rank == 0:
        _write_yaml(run_dir / "config.yaml", {
            "experiment": EXP_ID, "condition": condition, "condition_spec": spec,
            "dataset": dataset_key, "seed": seed, "debug": debug,
            "baseline_checkpoint": str(baseline_path),
            "baseline_checkpoint_stat": _checkpoint_stat(baseline_path),
            "run_signature": signature, "model_cfg": model_cfg,
            "calibration": EXP_4["calibration"], "spectral": EXP_4["spectral"],
            "training": EXP_4["training"], "evaluation": EXP_4["evaluation"],
            "spectral_basis_metadata": basis_payload["metadata"],
        })
    train_info = _train_condition(model, datasets, seed, condition, spec, signature,
                                  run_dir, device, ddp, logger, debug, max_len)
    if rank != 0:
        return None
    valid = train_info["valid"]
    test = train_info["test"]
    core = model.module if ddp else model
    if EXP_4["evaluation"]["compute_global_ner"]:
        ner = compute_ner_metrics(model, datasets, device, max_len, debug)
    else:
        active = _canonical_alphas(core.active_alphas.detach().cpu().tolist())
        ner = {"alpha_nodes": active, "s_pre": {"valid": None, "test": None},
               "pre_final": {"valid": [None] * len(active), "test": [None] * len(active)},
               "post_final": {"valid": [None] * len(active), "test": [None] * len(active)},
               "representation_mix": None}
    if EXP_4["evaluation"]["compute_weight_stats"]:
        weight_valid = _weight_stats_for_split(valid, datasets["valid"], max_len)
        weight_test = _weight_stats_for_split(test, datasets["test"], max_len)
    else:
        weight_valid = weight_test = None
    model_summary = {
        "kind": spec["kind"], "fusion": spec.get("fusion"),
        "alpha": spec.get("alpha"), "node_aux_loss_weight": spec.get("node_aux_loss_weight", 0.0),
        "n_params_total": int(sum(p.numel() for p in core.parameters())),
        "n_params_trainable": int(sum(p.numel() for p in core.parameters() if p.requires_grad)),
        "n_params_frozen": int(sum(p.numel() for p in core.parameters() if not p.requires_grad)),
        "train_seconds": train_info["train_seconds"],
        "epochs_run": train_info["epochs_run"], "best_epoch": train_info["best_epoch"],
        "best_valid_auc": train_info["best_valid_auc"],
        "micro_batch": train_info["micro_batch"], "accum_steps": train_info["accum_steps"],
        "peak_memory_mb": peak_memory_mb(device) if EXP_4["evaluation"]["compute_cost"] else None,
        "device": str(device),
        "inference": ({"test_seconds": test["elapsed"], "n_samples": test["n"],
                       "ms_per_sample": 1000.0 * test["elapsed"] / max(1, test["n"]),
                       "samples_per_sec": test["n"] / max(1e-9, test["elapsed"])}
                      if EXP_4["evaluation"]["compute_cost"] else None),
    }
    baseline_summary_path = (PATHS["results_root"] / "runs" / EXP_ID
                             / "baseline_frozen_prefix" / f"seed_{seed}" / "model_summary.json")
    if baseline_summary_path.exists() and model_summary["inference"]:
        base_ms = (json.loads(baseline_summary_path.read_text()).get("inference") or {}).get("ms_per_sample")
        if base_ms:
            model_summary["extra_latency_ratio"] = model_summary["inference"]["ms_per_sample"] / base_ms
    ckpt_path = run_dir / "checkpoint_best.pt"
    torch.save({"model_state": core.state_dict(), "condition": condition, "seed": seed,
                "kind": spec["kind"], "fusion": spec.get("fusion"), "alpha": spec.get("alpha"),
                "run_signature": signature, "baseline_checkpoint": str(baseline_path)},
               ckpt_path)
    node_metrics = {"valid": valid.get("node_metrics"), "test": test.get("node_metrics")}
    weighted_logit = {"valid": valid.get("weighted_logit_metrics"),
                      "test": test.get("weighted_logit_metrics")}
    valid_payload = {
        "experiment": EXP_ID, "condition": condition, "dataset": dataset_key,
        "seed": seed, "kind": spec["kind"], "fusion": spec.get("fusion"),
        "alpha": spec.get("alpha"), "node_aux_loss_weight": spec.get("node_aux_loss_weight", 0.0),
        "split": "valid", "primary": valid["primary_metrics"],
        "nodes": node_metrics["valid"], "weighted_logit": weighted_logit["valid"],
        "ner": _split_ner_payload(ner, "valid"),
        "weight_stats": weight_valid, "model_summary": model_summary,
        "training": {"best_epoch": train_info["best_epoch"],
                     "epochs_run": train_info["epochs_run"],
                     "best_valid_auc": train_info["best_valid_auc"]},
        "spectral_basis": {k: basis_payload["metadata"].get(k) for k in
                           ("calibration_split", "n_tokens", "n_samples", "token_dim",
                            "n_significant", "rank_relative_tolerance", "baseline_checkpoint",
                            "seed", "dataset")},
        "reference_ner": basis_payload["reference_ner"],
        "calibration_ner": basis_payload["calibration_ner"],
        "run_signature": signature,
    }
    test_payload = copy.deepcopy(valid_payload)
    test_payload["split"] = "test"
    test_payload["primary"] = test["primary_metrics"]
    test_payload["nodes"] = node_metrics["test"]
    test_payload["weighted_logit"] = weighted_logit["test"]
    test_payload["ner"] = _split_ner_payload(ner, "test")
    test_payload["weight_stats"] = weight_test
    per_seed = {
        "experiment": EXP_ID, "condition": condition, "dataset": dataset_key,
        "seed": seed, "kind": spec["kind"], "fusion": spec.get("fusion"),
        "alpha": spec.get("alpha"), "node_aux_loss_weight": spec.get("node_aux_loss_weight", 0.0),
        "variable": {"kind": spec["kind"], "fusion": spec.get("fusion"),
                     "alpha": spec.get("alpha"),
                     "node_aux_loss_weight": spec.get("node_aux_loss_weight", 0.0)},
        "valid_primary": valid["primary_metrics"], "test_primary": test["primary_metrics"],
        "node_metrics": node_metrics, "weighted_logit": weighted_logit,
        "ner": ner, "weight_stats": {"valid": weight_valid, "test": weight_test},
        "reference_ner": basis_payload["reference_ner"],
        "calibration_ner": basis_payload["calibration_ner"],
        "spectral_alpha_nodes": [float(a) for a in EXP_4["spectral"]["alpha_nodes"]],
        "spectral_basis_metadata": basis_payload["metadata"],
        "model_summary": model_summary,
        "training": {"best_epoch": train_info["best_epoch"],
                     "epochs_run": train_info["epochs_run"],
                     "best_valid_auc": train_info["best_valid_auc"]},
        "checkpoint": str(ckpt_path), "run_signature": signature,
    }
    _write_json(run_dir / "metrics_valid.json", valid_payload)
    _write_json(run_dir / "metrics_test.json", test_payload)
    _write_json(run_dir / "model_summary.json", model_summary)
    _write_json(run_dir / "alpha_weight_stats.json", {"valid": weight_valid, "test": weight_test})
    with open(run_dir / "alpha_node_metrics.csv", "w", newline="") as f:
        fields = ["split", "alpha", "auc", "logloss", "n"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in ("valid", "test"):
            for row in _node_rows(node_metrics[split]):
                writer.writerow({"split": split, "alpha": row["alpha"], "auc": row["auc"],
                                 "logloss": row["logloss"], "n": row["n"]})
    _write_json(run_dir / "metrics_combined.json", per_seed)
    _write_json(PATHS["metrics_root"] / EXP_ID / f"{condition}__{seed}.json", per_seed)
    logger.info(f"test_primary={test['primary_metrics']} "
                f"valid_primary={valid['primary_metrics']}")
    logger.info(f"ner_s_pre={ner['s_pre']} pre_final_valid={ner['pre_final']['valid']}")
    logger.info(f"model_summary={model_summary}")
    return per_seed


def _mean_std(values: List[float]) -> Dict[str, float]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(vals.mean()), "std": float(vals.std())}


def _load_runs() -> List[Dict]:
    metrics_dir = PATHS["metrics_root"] / EXP_ID
    runs = []
    for fp in sorted(metrics_dir.glob("*.json")):
        if fp.name in ("summary.json", "paired_deltas.json"):
            continue
        payload = json.loads(fp.read_text())
        if payload.get("experiment") == EXP_ID:
            runs.append(payload)
    return runs


def _aggregate_weight_stats(rs: List[Dict], split: str) -> Optional[Dict]:
    ws = [r["weight_stats"][split] for r in rs
          if r.get("weight_stats") and r["weight_stats"].get(split)]
    if not ws:
        return None
    means = np.asarray([w["overall"]["mean"] for w in ws], dtype=float)
    p10 = np.asarray([w["overall"]["p10"] for w in ws], dtype=float)
    p50 = np.asarray([w["overall"]["p50"] for w in ws], dtype=float)
    p90 = np.asarray([w["overall"]["p90"] for w in ws], dtype=float)
    within_std = np.asarray([w["overall"]["std"] for w in ws], dtype=float)
    entropy = np.asarray([w["overall"]["entropy"]["mean"] for w in ws], dtype=float)
    expected = np.asarray([w["overall"]["expected_alpha"]["mean"] for w in ws], dtype=float)
    by_len = defaultdict(lambda: {"n": [], "mean": []})
    for w in ws:
        for row in w.get("by_history_len", []):
            by_len[row["bucket"]]["n"].append(row["n"])
            by_len[row["bucket"]]["mean"].append(row["mean_weights"])
    by_group = defaultdict(lambda: {"n": [], "mean": []})
    for w in ws:
        for row in w.get("by_group", []):
            by_group[str(row["group"])]["n"].append(row["n"])
            by_group[str(row["group"])]["mean"].append(row["mean_weights"])
    return {
        "n_samples": int(np.mean([w["overall"]["n_samples"] for w in ws])),
        "mean": means.mean(axis=0).tolist(),
        "seed_std_of_mean": means.std(axis=0).tolist(),
        "within_sample_std": within_std.mean(axis=0).tolist(),
        "p10": p10.mean(axis=0).tolist(),
        "p50": p50.mean(axis=0).tolist(),
        "p90": p90.mean(axis=0).tolist(),
        "entropy_mean": float(entropy.mean()),
        "expected_alpha_mean": float(expected.mean()),
        "by_history_len": [
            {"bucket": b, "n": int(np.mean(v["n"])),
             "mean_weights": np.mean(np.asarray(v["mean"], dtype=float), axis=0).tolist()}
            for b, v in sorted(by_len.items())],
        "by_group": [
            {"group": g, "n": int(np.mean(v["n"])),
             "mean_weights": np.mean(np.asarray(v["mean"], dtype=float), axis=0).tolist()}
            for g, v in sorted(by_group.items())],
    }


def _aggregate_ner(rs: List[Dict]) -> Dict:
    first = rs[0]
    active = first["ner"]["alpha_nodes"]
    global_nodes = first.get("spectral_alpha_nodes") or EXP4_ALPHAS
    ref = [_mean_std([r["reference_ner"][i] for r in rs]) for i in range(len(global_nodes))]
    cal = [_mean_std([r["calibration_ner"][i] for r in rs]) for i in range(len(global_nodes))]
    pre_len = len(first["ner"]["pre_final"]["valid"]) if first["ner"].get("pre_final") else 0
    post_len = len(first["ner"]["post_final"]["valid"]) if first["ner"].get("post_final") else 0
    pre = {split: [_mean_std([r["ner"]["pre_final"][split][i] for r in rs])
                   for i in range(pre_len)] for split in ("valid", "test")}
    post = {split: [_mean_std([r["ner"]["post_final"][split][i] for r in rs])
                    for i in range(post_len)] for split in ("valid", "test")}
    mix = {}
    if any(r["ner"].get("representation_mix") for r in rs):
        for key in ("valid_pre", "valid_post", "test_pre", "test_post"):
            mix[key] = _mean_std([(r["ner"].get("representation_mix") or {}).get(key)
                                  for r in rs])
    return {"alpha_nodes": active, "spectral_alpha_nodes": global_nodes,
            "reference_ner": ref, "calibration_ner": cal,
            "s_pre": {split: _mean_std([r["ner"]["s_pre"][split] for r in rs])
                      for split in ("valid", "test")},
            "pre_final": pre, "post_final": post,
            "representation_mix": mix if mix else None}


def _aggregate_condition(rs: List[Dict]) -> Dict:
    first = rs[0]
    entry = {
        "n_seeds": len(rs), "seeds": sorted(int(r["seed"]) for r in rs),
        "kind": first.get("kind"), "fusion": first.get("fusion"),
        "alpha": first.get("alpha"),
        "node_aux_loss_weight": first.get("node_aux_loss_weight", 0.0),
        "variable": first.get("variable"),
        "valid_primary": {m: _mean_std([r["valid_primary"][m] for r in rs])
                          for m in ("auc", "logloss")},
        "test_primary": {m: _mean_std([r["test_primary"][m] for r in rs])
                         for m in ("auc", "logloss")},
        "node_metrics": {}, "weighted_logit": {},
        "ner": _aggregate_ner(rs),
        "weight_stats": {split: _aggregate_weight_stats(rs, split)
                         for split in ("valid", "test")},
    }
    for split in ("valid", "test"):
        keys = sorted({a for r in rs for a in (r.get("node_metrics", {}).get(split) or {})},
                      key=lambda x: float(x))
        entry["node_metrics"][split] = {
            a: {m: _mean_std([(r.get("node_metrics", {}).get(split) or {}).get(a, {}).get(m)
                              for r in rs]) for m in ("auc", "logloss")}
            for a in keys}
        entry["weighted_logit"][split] = {
            m: _mean_std([(r.get("weighted_logit", {}).get(split) or {}).get(m) for r in rs])
            for m in ("auc", "logloss")}
    for key in ("n_params_total", "n_params_trainable", "n_params_frozen",
                "train_seconds", "peak_memory_mb"):
        entry[key] = _mean_std([r.get("model_summary", {}).get(key) for r in rs])
    entry["inference_ms_per_sample"] = _mean_std(
        [((r.get("model_summary") or {}).get("inference") or {}).get("ms_per_sample")
         for r in rs])
    return entry


def _paired_deltas(runs: List[Dict], conditions: List[str]) -> List[Dict]:
    by_key = {(r["condition"], int(r["seed"])): r for r in runs}
    comparisons = []

    def add(comparison, baseline, scope, metric, getter_a, getter_b, seeds):
        deltas, a_vals, b_vals, used = [], [], [], []
        for seed in seeds:
            ra = by_key.get((comparison, seed))
            rb = by_key.get((baseline, seed))
            if ra is None or rb is None:
                continue
            a, b = getter_a(ra), getter_b(rb)
            if a is None or b is None:
                continue
            deltas.append(float(a) - float(b))
            a_vals.append(float(a))
            b_vals.append(float(b))
            used.append(seed)
        if not deltas:
            return
        seed_std = float(np.sqrt((np.var(a_vals) + np.var(b_vals)) / 2.0))
        mean_delta = float(np.mean(deltas))
        comparisons.append({
            "comparison": comparison, "baseline": baseline, "scope": scope,
            "metric": metric, "mean_delta": mean_delta,
            "std_delta": float(np.std(deltas)), "seed_std": seed_std,
            "uncertain": bool(len(deltas) < 2 or abs(mean_delta) <= seed_std),
            "n_pairs": len(deltas),
            "per_seed": {str(s): float(d) for s, d in zip(used, deltas)},
        })

    if "baseline_frozen_prefix" not in conditions:
        return comparisons
    baseline = "baseline_frozen_prefix"
    baseline_runs = [r for r in runs if r["condition"] == baseline]
    base_seeds = [int(r["seed"]) for r in baseline_runs]
    base_alpha_key = str(baseline_runs[0].get("alpha", 1.0))
    for cond in conditions:
        if cond == baseline:
            continue
        cond_runs = [r for r in runs if r["condition"] == cond]
        cond_seeds = [int(r["seed"]) for r in cond_runs]
        seeds = sorted(set(cond_seeds) & set(base_seeds))
        if not seeds:
            continue
        for split in ("valid", "test"):
            for metric in ("auc", "logloss"):
                add(cond, baseline, split, f"primary.{metric}",
                    lambda r, s=split, m=metric: r[f"{s}_primary"][m],
                    lambda r, s=split, m=metric: r[f"{s}_primary"][m], seeds)
            cond_alpha_keys = sorted((cond_runs[0].get("node_metrics", {}).get(split) or {}),
                                     key=lambda x: float(x))
            for alpha in cond_alpha_keys:
                for metric in ("auc", "logloss"):
                    add(cond, baseline, split, f"node.{alpha}.{metric}",
                        lambda r, a=alpha, s=split, m=metric: (
                            (r.get("node_metrics", {}).get(s) or {}).get(a, {}).get(m)),
                        lambda r, s=split, m=metric, a=base_alpha_key: (
                            (r.get("node_metrics", {}).get(s) or {}).get(a, {}).get(m)),
                        seeds)
            active = cond_runs[0]["ner"]["alpha_nodes"]
            n_pre = len(cond_runs[0]["ner"].get("pre_final", {}).get("valid", []))
            n_post = len(cond_runs[0]["ner"].get("post_final", {}).get("valid", []))
            for alpha_idx in range(n_pre):
                alpha = active[alpha_idx] if alpha_idx < len(active) else str(alpha_idx)
                for scope, key in (("valid", "valid"), ("test", "test")):
                    add(cond, baseline, scope, f"ner.pre_final.{alpha}",
                        lambda r, i=alpha_idx, s=key: r["ner"]["pre_final"][s][i],
                        lambda r, s=key: r["ner"]["pre_final"][s][0], seeds)
            for alpha_idx in range(n_post):
                alpha = active[alpha_idx] if alpha_idx < len(active) else str(alpha_idx)
                for scope, key in (("valid", "valid"), ("test", "test")):
                    add(cond, baseline, scope, f"ner.post_final.{alpha}",
                        lambda r, i=alpha_idx, s=key: r["ner"]["post_final"][s][i],
                        lambda r, s=key: r["ner"]["post_final"][s][0], seeds)
            if cond_runs[0]["ner"].get("representation_mix"):
                for split in ("valid", "test"):
                    add(cond, baseline, split, "ner.representation_mix.pre",
                        lambda r, s=split: (r["ner"].get("representation_mix") or {}).get(f"{s}_pre"),
                        lambda r, s=split: r["ner"]["pre_final"][s][0], seeds)
                    add(cond, baseline, split, "ner.representation_mix.post",
                        lambda r, s=split: (r["ner"].get("representation_mix") or {}).get(f"{s}_post"),
                        lambda r, s=split: r["ner"]["post_final"][s][0], seeds)
    # contextual-vs-baseline-fusion comparisons
    contexts = [("contextual_learned_expert_mix", "uniform_expert_mix"),
                ("contextual_learned_expert_mix", "global_learned_expert_mix"),
                ("contextual_learned_expert_mix", "contextual_representation_mix")]
    for comparison, baseline_name in contexts:
        comp_seeds = [int(r["seed"]) for r in runs if r["condition"] == comparison]
        base_fusion_seeds = [int(r["seed"]) for r in runs if r["condition"] == baseline_name]
        seeds = sorted(set(comp_seeds) & set(base_fusion_seeds))
        for split in ("valid", "test"):
            for metric in ("auc", "logloss"):
                add(comparison, baseline_name, split, f"primary.{metric}",
                    lambda r, s=split, m=metric: r[f"{s}_primary"][m],
                    lambda r, s=split, m=metric: r[f"{s}_primary"][m], seeds)
    return comparisons


def summarize_exp_4() -> Optional[Dict]:
    runs = _load_runs()
    if not runs:
        print(f"[summary] no runs under {PATHS['metrics_root'] / EXP_ID}")
        return None
    conditions = list(dict.fromkeys(r["condition"] for r in runs))
    summary = {cond: _aggregate_condition([r for r in runs if r["condition"] == cond])
               for cond in conditions}
    base_runs = {(int(r["seed"])): r for r in runs if r["condition"] == "baseline_frozen_prefix"}
    for cond, entry in summary.items():
        ratios = []
        for r in [x for x in runs if x["condition"] == cond]:
            base = base_runs.get(int(r["seed"]))
            if base is None:
                continue
            base_ms = ((base.get("model_summary") or {}).get("inference") or {}).get("ms_per_sample")
            own_ms = ((r.get("model_summary") or {}).get("inference") or {}).get("ms_per_sample")
            if base_ms and own_ms:
                ratios.append(float(own_ms) / float(base_ms))
        entry["extra_latency_ratio"] = _mean_std(ratios)
    paired = _paired_deltas(runs, conditions)
    alphas = runs[0].get("spectral_alpha_nodes") or EXP4_ALPHAS
    out = {"experiment": EXP_ID, "seeds": sorted({int(r["seed"]) for r in runs}),
           "conditions": conditions, "spectral_alpha_nodes": alphas,
           "quadrature_weights": EXP_4["spectral"]["quadrature_weights"],
           "summary": summary, "paired_deltas": paired}
    metrics_dir = PATHS["metrics_root"] / EXP_ID
    _write_json(metrics_dir / "summary.json", out)
    summary_alias = EXP_4.get("summary_alias", f"summary_{EXP_ID}")
    _write_json(PATHS["metrics_root"] / f"{summary_alias}.json", out)
    rows = []
    for cond, entry in summary.items():
        for split in ("valid", "test"):
            for metric in ("auc", "logloss"):
                stats = entry[f"{split}_primary"][metric]
                rows.append({"condition": cond, "scope": split, "alpha": "",
                             "metric": f"primary.{metric}", "mean": stats["mean"],
                             "std": stats["std"], "n_seeds": entry["n_seeds"]})
            for alpha, vals in entry["node_metrics"][split].items():
                for metric in ("auc", "logloss"):
                    rows.append({"condition": cond, "scope": split, "alpha": alpha,
                                 "metric": f"node.{metric}", "mean": vals[metric]["mean"],
                                 "std": vals[metric]["std"], "n_seeds": entry["n_seeds"]})
            for metric in ("auc", "logloss"):
                stats = (entry.get("weighted_logit", {}).get(split) or {}).get(metric)
                if stats and np.isfinite(stats["mean"]):
                    rows.append({"condition": cond, "scope": split, "alpha": "",
                                 "metric": f"weighted_logit.{metric}",
                                 "mean": stats["mean"], "std": stats["std"],
                                 "n_seeds": entry["n_seeds"]})
            stats = entry["weight_stats"][split]
            if stats:
                for i, alpha in enumerate(entry["ner"]["alpha_nodes"]):
                    rows.append({"condition": cond, "scope": split, "alpha": alpha,
                                 "metric": "weight.mean", "mean": stats["mean"][i],
                                 "std": stats["seed_std_of_mean"][i],
                                 "n_seeds": entry["n_seeds"]})
                rows.append({"condition": cond, "scope": split, "alpha": "",
                             "metric": "weight.entropy.mean", "mean": stats["entropy_mean"],
                             "std": float("nan"), "n_seeds": entry["n_seeds"]})
                rows.append({"condition": cond, "scope": split, "alpha": "",
                             "metric": "weight.expected_alpha.mean",
                             "mean": stats["expected_alpha_mean"], "std": float("nan"),
                             "n_seeds": entry["n_seeds"]})
            rm = entry["ner"].get("representation_mix")
            if rm:
                for key in (f"{split}_pre", f"{split}_post"):
                    stats = rm.get(key)
                    if stats:
                        rows.append({"condition": cond, "scope": split, "alpha": "",
                                     "metric": f"ner.representation_mix.{key.split('_')[-1]}",
                                     "mean": stats["mean"], "std": stats["std"],
                                     "n_seeds": entry["n_seeds"]})
        for alpha, stats in zip(entry["ner"]["spectral_alpha_nodes"], entry["ner"]["reference_ner"]):
            rows.append({"condition": cond, "scope": "reference", "alpha": alpha,
                         "metric": "ner.reference", "mean": stats["mean"],
                         "std": stats["std"], "n_seeds": entry["n_seeds"]})
        for alpha, stats in zip(entry["ner"]["spectral_alpha_nodes"], entry["ner"]["calibration_ner"]):
            rows.append({"condition": cond, "scope": "calibration", "alpha": alpha,
                         "metric": "ner.calibration", "mean": stats["mean"],
                         "std": stats["std"], "n_seeds": entry["n_seeds"]})
        for split in ("valid", "test"):
            rows.append({"condition": cond, "scope": split, "alpha": "",
                         "metric": "ner.s_pre",
                         "mean": entry["ner"]["s_pre"][split]["mean"],
                         "std": entry["ner"]["s_pre"][split]["std"],
                         "n_seeds": entry["n_seeds"]})
            for alpha, stats in zip(entry["ner"]["alpha_nodes"], entry["ner"]["pre_final"][split]):
                rows.append({"condition": cond, "scope": split, "alpha": alpha,
                             "metric": "ner.pre_final", "mean": stats["mean"],
                             "std": stats["std"], "n_seeds": entry["n_seeds"]})
            for alpha, stats in zip(entry["ner"]["alpha_nodes"], entry["ner"]["post_final"][split]):
                rows.append({"condition": cond, "scope": split, "alpha": alpha,
                             "metric": "ner.post_final", "mean": stats["mean"],
                             "std": stats["std"], "n_seeds": entry["n_seeds"]})
        for metric in ("n_params_total", "n_params_trainable", "n_params_frozen",
                       "train_seconds", "peak_memory_mb", "inference_ms_per_sample",
                       "extra_latency_ratio"):
            stats = entry.get(metric)
            if stats:
                rows.append({"condition": cond, "scope": "cost", "alpha": "",
                             "metric": metric, "mean": stats["mean"],
                             "std": stats["std"], "n_seeds": entry["n_seeds"]})
    with open(metrics_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "scope", "alpha", "metric",
                                               "mean", "std", "n_seeds"])
        writer.writeheader()
        writer.writerows(rows)
    with open(metrics_dir / "paired_deltas.csv", "w", newline="") as f:
        fields = ["comparison", "baseline", "scope", "metric", "mean_delta", "std_delta",
                  "seed_std", "uncertain", "n_pairs", "per_seed"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in paired:
            writer.writerow({**row, "per_seed": json.dumps(row["per_seed"])})
    _plot_exp4(summary, out)
    print(f"[done] {EXP_ID}: summary -> {metrics_dir / 'summary.json'}")
    return out


def _fixed_conditions(summary: Dict) -> List:
    rows = [(float(v["alpha"]), c, v) for c, v in summary.items()
            if v.get("kind") == "single" and v.get("alpha") is not None]
    return sorted(rows, key=lambda x: x[0])


def _plot_exp4(summary: Dict, out: Dict) -> None:
    fixed = _fixed_conditions(summary)
    if fixed and "baseline_frozen_prefix" in summary:
        base = summary["baseline_frozen_prefix"]
        series = {
            "reference": {"x": out["spectral_alpha_nodes"],
                          "y": [s["mean"] for s in base["ner"]["reference_ner"]],
                          "std": [s["std"] for s in base["ner"]["reference_ner"]]},
            "calibration": {"x": out["spectral_alpha_nodes"],
                            "y": [s["mean"] for s in base["ner"]["calibration_ner"]],
                            "std": [s["std"] for s in base["ner"]["calibration_ner"]]},
            "valid_transformed": {"x": [r[0] for r in fixed],
                                  "y": [r[2]["ner"]["pre_final"]["valid"][0]["mean"] for r in fixed],
                                  "std": [r[2]["ner"]["pre_final"]["valid"][0]["std"] for r in fixed]},
            "test_transformed": {"x": [r[0] for r in fixed],
                                 "y": [r[2]["ner"]["pre_final"]["test"][0]["mean"] for r in fixed],
                                 "std": [r[2]["ner"]["pre_final"]["test"][0]["std"] for r in fixed]},
        }
        figures.plot_series(series, "alpha", "Normalized Effective Rank",
                            "Alpha vs NER (pre-final S stream)",
                            PATHS["figures_root"] / f"{EXP_ID}_alpha_vs_ner.png")
        for metric in ("auc", "logloss"):
            s = {
                "valid": {"x": [r[0] for r in fixed],
                          "y": [r[2]["valid_primary"][metric]["mean"] for r in fixed],
                          "std": [r[2]["valid_primary"][metric]["std"] for r in fixed]},
                "test": {"x": [r[0] for r in fixed],
                         "y": [r[2]["test_primary"][metric]["mean"] for r in fixed],
                         "std": [r[2]["test_primary"][metric]["std"] for r in fixed]},
            }
            figures.plot_series(s, "alpha", metric.upper(),
                                f"Alpha vs {metric.upper()} (frozen prefix, fixed alpha)",
                                PATHS["figures_root"] / f"{EXP_ID}_alpha_vs_{metric}.png")
        base_pre = base["ner"]["s_pre"]["test"]["mean"]
        base_pre_std = base["ner"]["s_pre"]["test"]["std"]
        layer_series = {
            "s_pre (before transform)": {"x": [r[0] for r in fixed],
                                         "y": [base_pre for _ in fixed],
                                         "std": [base_pre_std for _ in fixed]},
            "after transform (pre-final)": {
                "x": [r[0] for r in fixed],
                "y": [r[2]["ner"]["pre_final"]["test"][0]["mean"] for r in fixed],
                "std": [r[2]["ner"]["pre_final"]["test"][0]["std"] for r in fixed]},
            "after final block": {
                "x": [r[0] for r in fixed],
                "y": [r[2]["ner"]["post_final"]["test"][0]["mean"] for r in fixed],
                "std": [r[2]["ner"]["post_final"]["test"][0]["std"] for r in fixed]},
        }
        figures.plot_series(layer_series, "alpha", "Normalized Effective Rank",
                            "NER at transform point / pre-final / post-final (test)",
                            PATHS["figures_root"] / f"{EXP_ID}_ner_layer_comparison.png")
    figures.plot_condition_metrics(summary, f"{EXP_ID}: condition comparison",
                                   PATHS["figures_root"] / f"{EXP_ID}_mixture_comparison.png")
    weight_series = {}
    for cond, entry in summary.items():
        if entry.get("kind") == "single" or entry.get("fusion") not in ("uniform", "global", "contextual"):
            continue
        stats = entry.get("weight_stats", {}).get("valid")
        if not stats:
            continue
        weight_series[cond] = {"x": entry["ner"]["alpha_nodes"], "mean": stats["mean"],
                               "p10": stats["p10"], "p90": stats["p90"]}
    if weight_series:
        figures.plot_weight_distribution(weight_series, f"{EXP_ID}: alpha weight distribution (valid)",
                                         PATHS["figures_root"] / f"{EXP_ID}_weight_distribution.png")


def run_experiment_4(dataset_key: str, seeds: List[int], device_arg: str = "auto",
                     debug: bool = False, skip_existing: bool = False,
                     conditions: Optional[List[str]] = None) -> None:
    ensure_dist()
    specs = resolve_conditions(conditions)
    for condition, spec in specs:
        for seed in seeds:
            run_condition_4(condition, spec, dataset_key, seed, device_arg=device_arg,
                            debug=debug, skip_existing=skip_existing)
    if dist_rank() == 0:
        summarize_exp_4()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_exp_4()


if __name__ == "__main__":
    main()
