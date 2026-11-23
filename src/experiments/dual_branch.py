"""Dual-branch OneTrans experiment with high/low-NER regularization (SPEC3).

Conditions:
- single_baseline: the original single OneTrans branch.
- dual_control: two independent OneTrans branches + three CTR heads, lambda=0.
- high_only / low_only / dual_split_ner: dual branches with the corresponding
  NER regularization weights.

Every run writes its own run_dir with config.yaml, metrics_by_epoch.csv,
loss_curves.csv, metrics_test.json, checkpoint_best_mix.pt and
model_summary.json. Multi-seed aggregation is written to
results/metrics/exp_3_dual_branch_ner/{summary.json,summary.csv,
paired_deltas.csv}.
"""

import contextlib
import copy
import csv
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

from config.dataset import CONFIG as DATA_CFG
from config.exp_3_dual_branch_ner import CONFIG as EXP_3
from config.onetrans import CONFIG as MODEL_CFG
from config.paths import CONFIG as PATHS
from config.sequence_probe import CONFIG as PROBE_CFG
from src.data.dataset import KuaiRandSequenceDataset, collate_batch
from src.experiments import figures
from src.experiments.diagnostics import (collect_states, compute_layer_ner,
                                         select_diagnostic_indices,
                                         select_ner_token_pairs, state_names)
from src.experiments.runner import (accum_steps_for, dist_rank, ensure_dist, get_data,
                                    MaskedOneTrans, peak_memory_mb, resolve_device)
from src.models.dual_branch_onetrans import DualBranchOneTrans
from src.models.ner_regularization import differentiable_ner_stats
from src.models.onetrans_diagnostic import OneTransDiagnostic
from src.utils.logging import setup_logger
from src.utils.metrics import auc_score, logloss_score
from src.utils.seed import set_seed

DUAL_PATHS = ["high", "low", "mix"]


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
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def _write_yaml(path: Path, payload) -> None:
    # JSON is a subset of YAML; avoids an extra runtime dependency.
    _write_json(path, payload)


def _fmt_lambda(value: float) -> str:
    return f"{value:g}"


def resolve_conditions(names: Optional[List[str]] = None, sweep: bool = False,
                       ner_lambda: Optional[float] = None) -> List:
    base = EXP_3["ner_lambda"] if ner_lambda is None else float(ner_lambda)
    specs = copy.deepcopy(EXP_3["conditions"])
    for spec in specs.values():
        for key in ("lambda_high", "lambda_low"):
            if spec.get(key) == "ner_lambda":
                spec[key] = base
            if float(spec.get(key, 0.0)) < 0:
                raise ValueError(f"lambda must be >= 0, got {key}={spec[key]}")
    if sweep:
        selected = [("single_baseline", specs["single_baseline"]),
                    ("dual_control", specs["dual_control"])]
        for lam in EXP_3["lambda_sweep"]:
            if lam <= 0:
                continue
            tag = _fmt_lambda(lam)
            selected += [
                (f"high_only_lam{tag}", {"mode": "dual", "lambda_high": lam, "lambda_low": 0.0}),
                (f"low_only_lam{tag}", {"mode": "dual", "lambda_high": 0.0, "lambda_low": lam}),
                (f"dual_split_ner_lam{tag}", {"mode": "dual", "lambda_high": lam, "lambda_low": lam}),
            ]
        return selected
    selected = names or EXP_3["default_conditions"]
    missing = [n for n in selected if n not in specs]
    if missing:
        raise ValueError(f"unknown conditions: {missing}")
    return [(n, specs[n]) for n in selected]


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


def _layer_fields(prefix: str, num_layers: int) -> List[str]:
    return [f"ner_{prefix}_input"] + [f"ner_{prefix}_l{i}" for i in range(1, num_layers + 1)]


def _ner_fields(ner: Dict[str, float], prefix: str) -> Dict[str, float]:
    out = {}
    for name, value in ner.items():
        suffix = "input" if name == "backbone_input" else name.replace("layer_", "l")
        out[f"ner_{prefix}_{suffix}"] = value
    return out


def epoch_fields(num_layers: int) -> List[str]:
    fields = ["epoch", "step", "seed", "condition",
              "loss_ctr_high", "loss_ctr_low", "loss_ctr_mix", "loss_ctr",
              "loss_ner_high", "loss_ner_low",
              "loss_ner_high_weighted", "loss_ner_low_weighted",
              "lambda_high", "lambda_low", "loss_total", "grad_norm",
              "ner_high_observed", "ner_low_observed",
              "auc_high", "logloss_high", "auc_low", "logloss_low",
              "auc_mix", "logloss_mix", "auc_avg", "logloss_avg",
              "auc_single", "logloss_single"]
    for prefix in ("high", "low", "single"):
        fields += _layer_fields(prefix, num_layers)
    return fields


STEP_FIELDS = [
    "epoch", "step", "micro_step", "seed", "condition",
    "loss_ctr_high", "loss_ctr_low", "loss_ctr_mix", "loss_ctr",
    "loss_ner_high", "loss_ner_low",
    "loss_ner_high_weighted", "loss_ner_low_weighted",
    "lambda_high_effective", "lambda_low_effective", "loss_total",
    "grad_norm_total", "grad_norm_high", "grad_norm_low", "grad_norm_mix",
    "ner_high_observed", "ner_low_observed", "h_norm_high", "h_norm_low",
    "ner_high_tokens", "ner_low_tokens",
]


def _path_metrics(result: Dict) -> Dict:
    y_true = result["y_true"]
    out = {}
    for path, pred in result["paths"].items():
        out[path] = {"auc": auc_score(y_true, pred),
                     "logloss": logloss_score(y_true, pred),
                     "n": int(len(y_true))}
    return out


@torch.no_grad()
def evaluate_paths(model, loader, device, mode: str) -> Dict:
    core = model.module if hasattr(model, "module") else model
    core.eval()
    paths = (["single"] if mode == "single"
             else DUAL_PATHS + (["avg"] if EXP_3["evaluation"]["report_logit_avg"] else []))
    y_true, preds = [], defaultdict(list)
    t0 = time.time()
    n_samples = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = core(batch)
        for path in paths:
            if mode == "single":
                preds[path].append(out["pred"].float().cpu().numpy())
            else:
                preds[path].append(out[f"pred_{path}"].float().cpu().numpy())
        y_true.append(batch["label"].cpu().numpy())
        n_samples += int(batch["label"].numel())
    return {"y_true": np.concatenate(y_true),
            "paths": {k: np.concatenate(v) for k, v in preds.items()},
            "elapsed": time.time() - t0, "n": n_samples}


def _diagnostic_setup(dataset: KuaiRandSequenceDataset, n_samples: int, max_tokens: int,
                      max_len: int):
    idx = select_diagnostic_indices(dataset, n_samples, PROBE_CFG["diagnostic_seed"])
    lengths = np.minimum(dataset.position[idx], max_len)
    rank_arr, pos_arr = select_ner_token_pairs(lengths, max_tokens, PROBE_CFG["ner_seed"])
    return idx, rank_arr, pos_arr


@torch.no_grad()
def collect_dual_token_states(model, dataset: KuaiRandSequenceDataset,
                              sample_indices: np.ndarray, rank_arr: np.ndarray,
                              pos_arr: np.ndarray, batch_size: int, device,
                              token_dim: int, num_states: int):
    core = model.module if hasattr(model, "module") else model
    core.eval()
    loader = DataLoader(Subset(dataset, sample_indices.tolist()), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_batch)
    chunks_high = chunks_low = None
    consumed = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = core(batch, return_sequence_states=True)
        states_high = out["states_high"]
        states_low = out["states_low"]
        if len(states_high) != num_states or len(states_low) != num_states:
            raise RuntimeError("sequence-state count does not match state_names")
        b = states_high[0].shape[0]
        if chunks_high is None:
            chunks_high = [[] for _ in range(num_states)]
            chunks_low = [[] for _ in range(num_states)]
        sel = (rank_arr >= consumed) & (rank_arr < consumed + b)
        ranks_l = rank_arr[sel] - consumed
        poss_l = pos_arr[sel]
        for chunks, states in ((chunks_high, states_high), (chunks_low, states_low)):
            for li, state in enumerate(states):
                state = state.float()
                flat = state.view(-1, state.shape[-1])[ranks_l * state.shape[1] + poss_l]
                if flat.shape[0]:
                    chunks[li].append(flat.cpu())
        consumed += b
    if chunks_high is None:
        return ([torch.zeros(0, token_dim) for _ in range(num_states)],
                [torch.zeros(0, token_dim) for _ in range(num_states)])
    tokens_high = [torch.cat(ch) if ch else torch.zeros(0, token_dim) for ch in chunks_high]
    tokens_low = [torch.cat(ch) if ch else torch.zeros(0, token_dim) for ch in chunks_low]
    return tokens_high, tokens_low


def eval_ner(model, dataset, diag, names, batch_size, device, mode, token_dim):
    idx, rank_arr, pos_arr = diag
    if mode == "single":
        states = collect_states(model, dataset, idx, rank_arr, pos_arr, batch_size, device,
                                pool_last_layer=False)
        return {"single": compute_layer_ner(states["tokens"], names)}
    tokens_high, tokens_low = collect_dual_token_states(
        model, dataset, idx, rank_arr, pos_arr, batch_size, device, token_dim, len(names))
    return {"high": compute_layer_ner(tokens_high, names),
            "low": compute_layer_ner(tokens_low, names)}


def _grad_norms(groups: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    sq = {name: 0.0 for name in groups if name != "total"}
    for name, params in groups.items():
        if name == "total":
            continue
        for p in params:
            if p.grad is not None:
                sq[name] += float(p.grad.detach().float().pow(2).sum())
    out = {name: math.sqrt(value) for name, value in sq.items()}
    out["total"] = math.sqrt(sum(value for value in sq.values()))
    return out


def _branch_params(core, mode):
    base = core.model if isinstance(core, MaskedOneTrans) else core
    if mode == "single":
        all_params = list(base.parameters())
        return {"single": all_params, "total": all_params}
    return {"high": list(base.high_branch.parameters()),
            "low": list(base.low_branch.parameters()),
            "mix": list(base.mix_head.parameters()),
            "total": list(base.parameters())}


def run_condition_3(condition: str, spec: Dict, dataset_key: str, seed: int,
                    device_arg: str = "auto", debug: bool = False,
                    skip_existing: bool = False) -> Optional[Dict]:
    ensure_dist()
    ddp = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    rank = torch.distributed.get_rank() if ddp else 0
    mode = spec["mode"]
    exp_id = EXP_3["experiment"]
    run_dir = PATHS["results_root"] / "runs" / exp_id / condition / f"seed_{seed}"
    metrics_fp = run_dir / "metrics_test.json"
    if skip_existing and metrics_fp.exists():
        if rank == 0:
            print(f"[skip] {metrics_fp} already exists, loading it")
            loaded = json.loads(metrics_fp.read_text())
            per_seed = PATHS["metrics_root"] / EXP_3["experiment"] / f"{condition}__{seed}.json"
            if not per_seed.exists():
                _write_json(per_seed, loaded)
            return loaded
        return None

    if ddp:
        device = (torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
                  if torch.cuda.is_available() else torch.device("cpu"))
    else:
        device = resolve_device(device_arg)
    logger = (setup_logger(f"{exp_id}_{condition}_{seed}",
                           PATHS["logs_root"] / exp_id / condition / f"{seed}.log")
              if rank == 0 else None)
    if logger:
        logger.info(f"experiment={exp_id} condition={condition} mode={mode} "
                    f"dataset={dataset_key} seed={seed} device={device} ddp={ddp}")

    set_seed(seed)
    data = get_data(dataset_key)
    max_len = MODEL_CFG["max_history_len"]
    dbg = EXP_3["debug"] if debug else {}
    datasets = {
        split: KuaiRandSequenceDataset(data, split, max_len,
                                       max_rows=dbg.get(f"{split}_samples"))
        for split in ("train", "valid", "test")
    }
    if logger:
        logger.info("samples: " + ", ".join(f"{s}={len(d)}" for s, d in datasets.items()))

    model_cfg = {**MODEL_CFG, "mix_head": EXP_3["mix_head"], "use_residual_gate": False}
    if mode == "single":
        base = OneTransDiagnostic(data.vocab_sizes(), model_cfg).to(device)
    else:
        base = DualBranchOneTrans(data.vocab_sizes(), model_cfg).to(device)
    model = MaskedOneTrans(base, None)
    if ddp:
        model = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
                 if torch.cuda.is_available() else torch.nn.parallel.DistributedDataParallel(model))

    world_size = torch.distributed.get_world_size() if ddp else 1
    accum_multiplier = (1 if mode == "single"
                        else int(EXP_3["training"]["accum_steps_multiplier"]))
    accum_steps = max(1, accum_steps_for(max_len, MODEL_CFG["batch_size"], device)
                      * accum_multiplier)
    micro_batch = max(1, MODEL_CFG["batch_size"] // (accum_steps * world_size))
    eval_bs = max(256, MODEL_CFG["eval_batch_size"] * MODEL_CFG["max_history_len"] // max_len)
    ner_bs = max(256, min(eval_bs, 1024))

    train_sampler = (torch.utils.data.DistributedSampler(datasets["train"], shuffle=True, seed=seed)
                     if ddp else None)
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
    if logger and (accum_steps > 1 or world_size > 1):
        logger.info(f"effective batch={MODEL_CFG['batch_size']}: micro_batch={micro_batch} x "
                    f"{accum_steps} accumulation x {world_size} ranks")

    names = state_names(MODEL_CFG["num_layers"], False)
    n_diag = dbg.get("diagnostic_samples", PROBE_CFG["diagnostic_samples"])
    n_tokens = dbg.get("ner_max_tokens", PROBE_CFG["ner_max_tokens"])
    valid_diag = _diagnostic_setup(datasets["valid"], n_diag, n_tokens, max_len)
    test_diag = _diagnostic_setup(datasets["test"], n_diag, n_tokens, max_len)

    optimizer = torch.optim.Adam(
        (model.module if ddp else model).param_groups(
            MODEL_CFG["dense_learning_rate"], MODEL_CFG["sparse_learning_rate"]),
        weight_decay=MODEL_CFG["weight_decay"])
    params_by_branch = _branch_params(model.module if ddp else model, mode)
    amp_dtype = {"none": None, "bf16": torch.bfloat16}.get(MODEL_CFG["amp_dtype"])
    n_epochs = dbg.get("epochs") or MODEL_CFG["epochs"]
    total_optim_steps = max(1, math.ceil(len(train_loader) / accum_steps) * n_epochs)
    warmup_steps = max(1, int(total_optim_steps * float(EXP_3["ner"]["warmup_ratio"])))
    epsilon = float(EXP_3["ner"]["epsilon"])
    backend = EXP_3["ner"]["spectral_backend"]
    if list(EXP_3["ner_target_layers"]) != ["last"]:
        raise ValueError("phase 1 NER regularization supports only ['last']")

    fields = epoch_fields(MODEL_CFG["num_layers"])
    epoch_logger = step_logger = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(run_dir / "config.yaml", {
            "experiment": exp_id, "condition": condition, "seed": seed,
            "dataset": dataset_key, "mode": mode, "condition_spec": spec,
            "variable": {
                "mode": mode, "lambda_high": spec.get("lambda_high", 0.0),
                "lambda_low": spec.get("lambda_low", 0.0)},
            "model": model_cfg, "training": {
                "effective_batch_size": MODEL_CFG["batch_size"],
                "accum_steps": accum_steps, "epochs": n_epochs,
                "warmup_steps": warmup_steps, "warmup_ratio": EXP_3["ner"]["warmup_ratio"],
                "debug": debug, "debug_caps": dbg},
            "ner": EXP_3["ner"], "ner_target_layers": EXP_3["ner_target_layers"],
            "probe": PROBE_CFG, "exp_config": EXP_3,
        })
        epoch_logger = CsvLogger(run_dir / "metrics_by_epoch.csv", fields)
        step_logger = CsvLogger(run_dir / "loss_curves.csv", STEP_FIELDS)

    train_start = time.time()
    best_auc, best_epoch, best_state = -1.0, -1, None
    patience_left = MODEL_CFG["early_stop_patience"]
    epochs_run = 0
    global_step = 0
    micro_step = 0

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
            warmup_scale = min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0
            lam_high = float(spec.get("lambda_high", 0.0)) * warmup_scale
            lam_low = float(spec.get("lambda_low", 0.0)) * warmup_scale

            with sync_ctx:
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        out = model(batch)
                        if mode == "single":
                            bce_single = F.binary_cross_entropy_with_logits(
                                out["logit"].float(), batch["label"])
                        else:
                            bce_high = F.binary_cross_entropy_with_logits(
                                out["logit_high"].float(), batch["label"])
                            bce_low = F.binary_cross_entropy_with_logits(
                                out["logit_low"].float(), batch["label"])
                            bce_mix = F.binary_cross_entropy_with_logits(
                                out["logit_mix"].float(), batch["label"])
                else:
                    out = model(batch)
                    if mode == "single":
                        bce_single = F.binary_cross_entropy_with_logits(
                            out["logit"], batch["label"])
                    else:
                        bce_high = F.binary_cross_entropy_with_logits(
                            out["logit_high"], batch["label"])
                        bce_low = F.binary_cross_entropy_with_logits(
                            out["logit_low"], batch["label"])
                        bce_mix = F.binary_cross_entropy_with_logits(
                            out["logit_mix"], batch["label"])

                if mode == "single":
                    loss_ctr = bce_single
                    loss_ner_h = loss_ner_l = torch.zeros((), device=device)
                    loss_ner = torch.zeros((), device=device)
                    stats_h = stats_l = None
                else:
                    loss_ctr = (bce_high + bce_low + bce_mix) / 3.0
                    if lam_high > 0:
                        stats_h = differentiable_ner_stats(
                            out["last_state_high"], out["hist_mask"], epsilon, backend)
                        loss_ner_h = -stats_h["h_norm"]
                    else:
                        with torch.no_grad():
                            stats_h = differentiable_ner_stats(
                                out["last_state_high"], out["hist_mask"], epsilon, backend)
                        loss_ner_h = -stats_h["h_norm"].detach()
                    if lam_low > 0:
                        stats_l = differentiable_ner_stats(
                            out["last_state_low"], out["hist_mask"], epsilon, backend)
                        loss_ner_l = stats_l["h_norm"]
                    else:
                        with torch.no_grad():
                            stats_l = differentiable_ner_stats(
                                out["last_state_low"], out["hist_mask"], epsilon, backend)
                        loss_ner_l = stats_l["h_norm"].detach()
                    loss_ner = lam_high * loss_ner_h + lam_low * loss_ner_l
                loss = loss_ctr + loss_ner
                (loss / accum_steps).backward()

            if is_update:
                norms = _grad_norms(params_by_branch)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_norms.append(norms["total"])
                global_step += 1
                grad_values = (norms["total"], norms.get("high", norms["total"]),
                               norms.get("low", 0.0), norms.get("mix", 0.0))
            else:
                grad_values = ("", "", "", "")

            row = {
                "epoch": epoch + 1, "step": global_step, "micro_step": micro_step,
                "seed": seed, "condition": condition,
                "loss_ctr": float(loss_ctr.detach()),
                "loss_total": float(loss.detach()),
                "lambda_high_effective": lam_high, "lambda_low_effective": lam_low,
                "grad_norm_total": grad_values[0], "grad_norm_high": grad_values[1],
                "grad_norm_low": grad_values[2], "grad_norm_mix": grad_values[3],
            }
            if mode == "single":
                row.update({"loss_ctr_high": "", "loss_ctr_low": "", "loss_ctr_mix": "",
                            "loss_ner_high": "", "loss_ner_low": "",
                            "loss_ner_high_weighted": "", "loss_ner_low_weighted": "",
                            "ner_high_observed": "", "ner_low_observed": "",
                            "h_norm_high": "", "h_norm_low": "",
                            "ner_high_tokens": "", "ner_low_tokens": ""})
            else:
                row.update({
                    "loss_ctr_high": float(bce_high.detach()),
                    "loss_ctr_low": float(bce_low.detach()),
                    "loss_ctr_mix": float(bce_mix.detach()),
                    "loss_ner_high": float(loss_ner_h.detach()),
                    "loss_ner_low": float(loss_ner_l.detach()),
                    "loss_ner_high_weighted": float((lam_high * loss_ner_h).detach()),
                    "loss_ner_low_weighted": float((lam_low * loss_ner_l).detach()),
                    "ner_high_observed": float(stats_h["ner"].detach()),
                    "ner_low_observed": float(stats_l["ner"].detach()),
                    "h_norm_high": float(stats_h["h_norm"].detach()),
                    "h_norm_low": float(stats_l["h_norm"].detach()),
                    "ner_high_tokens": stats_h["n_tokens"],
                    "ner_low_tokens": stats_l["n_tokens"],
                })
            for key, value in row.items():
                if key in ("loss_ctr", "loss_total", "loss_ctr_high", "loss_ctr_low",
                           "loss_ctr_mix", "loss_ner_high", "loss_ner_low",
                           "loss_ner_high_weighted", "loss_ner_low_weighted",
                           "ner_high_observed", "ner_low_observed",
                           "h_norm_high", "h_norm_low"):
                    if value != "":
                        sums[key] += value
            if rank == 0:
                step_logger.write(row)
            n_micro += 1
            micro_step += 1

        epochs_run = epoch + 1
        mean_losses = {k: (v / n_micro if n_micro else float("nan")) for k, v in sums.items()}
        mean_grad = float(np.mean(grad_norms)) if grad_norms else float("nan")

        if rank == 0:
            res_valid = evaluate_paths(model, valid_loader, device, mode)
            valid_metrics = _path_metrics(res_valid)
            ner_valid = eval_ner(model, datasets["valid"], valid_diag, names, ner_bs,
                                 device, mode, base.token_dim)
            primary = "mix" if mode == "dual" else "single"
            auc_primary = valid_metrics[primary]["auc"]
            row = {
                "epoch": epoch + 1, "step": global_step, "seed": seed, "condition": condition,
                "loss_ctr": mean_losses.get("loss_ctr"),
                "loss_total": mean_losses.get("loss_total"),
                "lambda_high": float(spec.get("lambda_high", 0.0)),
                "lambda_low": float(spec.get("lambda_low", 0.0)),
                "grad_norm": mean_grad,
                "loss_ctr_high": mean_losses.get("loss_ctr_high", ""),
                "loss_ctr_low": mean_losses.get("loss_ctr_low", ""),
                "loss_ctr_mix": mean_losses.get("loss_ctr_mix", ""),
                "loss_ner_high": mean_losses.get("loss_ner_high", ""),
                "loss_ner_low": mean_losses.get("loss_ner_low", ""),
                "loss_ner_high_weighted": mean_losses.get("loss_ner_high_weighted", ""),
                "loss_ner_low_weighted": mean_losses.get("loss_ner_low_weighted", ""),
                "ner_high_observed": mean_losses.get("ner_high_observed", ""),
                "ner_low_observed": mean_losses.get("ner_low_observed", ""),
            }
            for path, metrics in valid_metrics.items():
                row[f"auc_{path}"] = metrics["auc"]
                row[f"logloss_{path}"] = metrics["logloss"]
            for branch, layers in ner_valid.items():
                row.update(_ner_fields(layers, branch))
            epoch_logger.write(row)
            epoch_logger.flush()
            logger.info(f"epoch {epoch + 1}: valid_auc_{primary}={auc_primary:.6f} "
                        f"train_loss={mean_losses.get('loss_ctr', float('nan')):.6f}")
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
                logger.info(f"early stop at epoch {epoch + 1} "
                            f"(best epoch {best_epoch}, auc={best_auc:.6f})")
            break

    train_seconds = time.time() - train_start
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if rank != 0:
        return None

    if epoch_logger is not None:
        epoch_logger.close()
        step_logger.close()

    valid_res = evaluate_paths(model, valid_loader, device, mode)
    test_res = evaluate_paths(model, test_loader, device, mode)
    valid_metrics = _path_metrics(valid_res)
    test_metrics = _path_metrics(test_res)
    ner_valid = eval_ner(model, datasets["valid"], valid_diag, names, ner_bs,
                         device, mode, base.token_dim)
    ner_test = (eval_ner(model, datasets["test"], test_diag, names, ner_bs,
                         device, mode, base.token_dim)
                if EXP_3["evaluation"]["compute_test_ner"] else None)

    core = model.module if ddp else model
    base_model = core.model if isinstance(core, MaskedOneTrans) else core
    n_params = {name: int(sum(p.numel() for p in params))
                for name, params in params_by_branch.items()}
    inference = {
        "test_seconds": test_res["elapsed"],
        "n_samples": test_res["n"],
        "ms_per_sample": 1000.0 * test_res["elapsed"] / max(1, test_res["n"]),
        "samples_per_sec": test_res["n"] / max(1e-9, test_res["elapsed"]),
    }
    branch_config = {k: MODEL_CFG[k] for k in
                     ("embedding_dim", "token_dim", "num_layers", "num_heads",
                      "num_ns_token", "tokenizer_type", "reduction_ratio",
                      "tower_hidden_units", "net_dropout", "max_history_len")}
    model_summary = {
        "mode": mode,
        "token_dim": MODEL_CFG["token_dim"],
        "num_ns_token": MODEL_CFG["num_ns_token"],
        "readout_dim": getattr(base_model, "readout_dim", None) or base_model.token_dim * base_model.num_ns_token,
        "num_layers": MODEL_CFG["num_layers"],
        "num_heads": MODEL_CFG["num_heads"],
        "branch_config": branch_config,
        "mix_head": EXP_3["mix_head"],
        "n_params_total": n_params["total"],
        "n_params_high": n_params.get("high"),
        "n_params_low": n_params.get("low"),
        "n_params_mix": n_params.get("mix"),
        "shared_parameters": 0 if mode == "dual" else None,
        "train_seconds": train_seconds,
        "epochs_run": epochs_run,
        "peak_memory_mb": peak_memory_mb(device),
        "device": str(device),
        "inference": inference,
    }
    ckpt_path = run_dir / "checkpoint_best_mix.pt"
    torch.save({"model_state": base.state_dict(), "condition": condition, "seed": seed,
                "mode": mode, "experiment": exp_id,
                "lambda_high": spec.get("lambda_high", 0.0),
                "lambda_low": spec.get("lambda_low", 0.0)}, ckpt_path)

    variable = {"mode": mode, "lambda_high": float(spec.get("lambda_high", 0.0)),
                "lambda_low": float(spec.get("lambda_low", 0.0))}
    metrics = {
        "experiment": exp_id,
        "condition": condition,
        "dataset": dataset_key,
        "dataset_version": DATA_CFG["dataset_names"].get(dataset_key, dataset_key),
        "mode": mode,
        "seed": seed,
        "variable": variable,
        "sample_counts": {s: len(d) for s, d in datasets.items()},
        "training": {"best_valid_auc_primary": best_auc, "best_epoch": best_epoch,
                     "epochs_run": epochs_run, "train_seconds": train_seconds,
                     "effective_batch_size": MODEL_CFG["batch_size"],
                     "accum_steps": accum_steps,
                     "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                     "warmup_steps": warmup_steps,
                     "debug": debug},
        "valid": valid_metrics,
        "test": test_metrics,
        "ner_valid": ner_valid,
        "ner_test": ner_test,
        "diagnostic_samples": int(len(valid_diag[0])),
        "ner_tokens": int(len(valid_diag[1])),
        "model_summary": model_summary,
        "checkpoint": str(ckpt_path),
    }
    _write_json(run_dir / "metrics_test.json", metrics)
    _write_json(run_dir / "model_summary.json", model_summary)
    metrics_dir = PATHS["metrics_root"] / exp_id
    _write_json(metrics_dir / f"{condition}__{seed}.json", metrics)
    test_msg = {k: {m: round(v, 6) for m, v in vals.items() if isinstance(v, float)}
                for k, vals in test_metrics.items()}
    logger.info(f"test={test_msg}")
    logger.info(f"model_summary={model_summary}")
    return metrics


def _load_runs(exp_id: str) -> List[Dict]:
    metrics_dir = PATHS["metrics_root"] / exp_id
    runs = []
    for fp in sorted(metrics_dir.glob("*.json")):
        if fp.name == "summary.json":
            continue
        payload = json.loads(fp.read_text())
        if payload.get("experiment") == exp_id:
            runs.append(payload)
    return runs


def _mean_std(values: List[float]) -> Dict[str, float]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(vals.mean()), "std": float(vals.std())}


def _paired_deltas(runs: List[Dict], conditions: List[str]) -> List[Dict]:
    by_key = {(r["condition"], int(r["seed"])): r for r in runs}
    seeds = sorted({int(r["seed"]) for r in runs})
    comparisons = []

    def add(comparison, baseline, scope, metric, getter_a, getter_b, seeds_sel):
        deltas, a_vals, b_vals, used_seeds = [], [], [], []
        for seed in seeds_sel:
            ra = by_key.get((comparison, seed))
            rb = by_key.get((baseline, seed))
            if ra is None or rb is None:
                continue
            a = getter_a(ra)
            b = getter_b(rb)
            if a is None or b is None:
                continue
            deltas.append(float(a) - float(b))
            a_vals.append(float(a))
            b_vals.append(float(b))
            used_seeds.append(seed)
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
            "per_seed": {str(s): float(d) for s, d in zip(used_seeds, deltas)},
        })

    for cond in [c for c in conditions if c != "single_baseline"]:
        cond_run = next((r for r in runs if r["condition"] == cond), None)
        if cond_run is None or cond_run.get("mode") != "dual":
            continue
        cond_seeds = [int(r["seed"]) for r in runs if r["condition"] == cond]
        for baseline in ("single_baseline", "dual_control"):
            if baseline == cond:
                continue
            base_seeds = [int(r["seed"]) for r in runs if r["condition"] == baseline]
            paired_seeds = sorted(set(cond_seeds) & set(base_seeds))
            if not paired_seeds:
                continue
            for path in DUAL_PATHS:
                base_path = "single" if baseline == "single_baseline" else path
                for split in ("valid", "test"):
                    for metric in ("auc", "logloss"):
                        add(cond, baseline, split, f"{path}.{metric}",
                            lambda r, p=path, s=split, m=metric: r[s][p][m],
                            lambda r, s=split, m=metric, p=base_path: r[s][p][m],
                            paired_seeds)
            for branch in ("high", "low"):
                for split, ner_key in (("valid", "ner_valid"), ("test", "ner_test")):
                    layers = (cond_run.get(ner_key) or {}).get(branch, {})
                    for layer in layers:
                        add(cond, baseline, split, f"ner.{branch}.{layer}",
                            lambda r, b=branch, l=layer, k=ner_key: r[k][b][l],
                            lambda r, b=branch, l=layer, k=ner_key: (
                                r[k]["single"][l] if r.get("mode") == "single"
                                else r[k][b][l]),
                            paired_seeds)
    return comparisons


def summarize_exp3(exp_id: Optional[str] = None) -> Optional[Dict]:
    exp_id = exp_id or EXP_3["experiment"]
    runs = _load_runs(exp_id)
    if not runs:
        print(f"[summary] no runs under {PATHS['metrics_root'] / exp_id}")
        return None
    conditions = list(dict.fromkeys(r["condition"] for r in runs))
    summary = {}
    for cond in conditions:
        rs = [r for r in runs if r["condition"] == cond]
        entry = {"n_seeds": len(rs), "seeds": sorted(int(r["seed"]) for r in rs),
                 "mode": rs[0]["mode"], "variable": rs[0].get("variable")}
        for split in ("valid", "test"):
            paths = sorted({p for r in rs for p in r[split]})
            entry[split] = {path: {metric: _mean_std([r[split][path][metric] for r in rs])
                                   for metric in ("auc", "logloss")}
                            for path in paths}
        for ner_key in ("ner_valid", "ner_test"):
            branches = sorted({b for r in rs if r.get(ner_key) for b in r[ner_key]})
            entry[ner_key] = {}
            for branch in branches:
                layers = list(next(r[ner_key][branch] for r in rs if r.get(ner_key)
                                   and branch in r[ner_key]))
                entry[ner_key][branch] = {
                    layer: _mean_std([r[ner_key][branch][layer] for r in rs
                                      if r.get(ner_key) and branch in r[ner_key]])
                    for layer in layers}
        for key in ("train_seconds", "peak_memory_mb", "n_params_total"):
            vals = []
            for r in rs:
                value = r.get("model_summary", {}).get(key)
                vals.append(value)
            entry[key] = _mean_std(vals)
        entry["inference_ms_per_sample"] = _mean_std(
            [r.get("model_summary", {}).get("inference", {}).get("ms_per_sample") for r in rs])
        summary[cond] = entry

    paired = (_paired_deltas(runs, conditions)
              if EXP_3["analysis"].get("compute_paired_deltas", True) else [])
    out = {"experiment": exp_id, "seeds": sorted({int(r["seed"]) for r in runs}),
           "conditions": conditions, "summary": summary,
           "paired_deltas": paired}
    metrics_dir = PATHS["metrics_root"] / exp_id
    _write_json(metrics_dir / "summary.json", out)

    rows = []
    for cond, entry in summary.items():
        for split in ("valid", "test"):
            for path, metrics in entry[split].items():
                for metric, stats in metrics.items():
                    rows.append({"condition": cond, "scope": split,
                                 "metric": f"{path}.{metric}",
                                 "mean": stats["mean"], "std": stats["std"],
                                 "n_seeds": entry["n_seeds"]})
        for ner_key in ("ner_valid", "ner_test"):
            for branch, layers in entry[ner_key].items():
                for layer, stats in layers.items():
                    rows.append({"condition": cond, "scope": ner_key,
                                 "metric": f"ner.{branch}.{layer}",
                                 "mean": stats["mean"], "std": stats["std"],
                                 "n_seeds": entry["n_seeds"]})
    with open(metrics_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "scope", "metric", "mean",
                                               "std", "n_seeds"])
        writer.writeheader()
        writer.writerows(rows)
    with open(metrics_dir / "paired_deltas.csv", "w", newline="") as f:
        fields = ["comparison", "baseline", "scope", "metric", "mean_delta", "std_delta",
                  "seed_std", "uncertain", "n_pairs", "per_seed"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in paired:
            writer.writerow({**row, "per_seed": json.dumps(row["per_seed"])})

    ner_by_cond = {}
    for cond, entry in summary.items():
        for branch, layers in entry["ner_test"].items():
            ner_by_cond[f"{cond}/{branch}"] = layers
    if ner_by_cond:
        figures.plot_ner_curves(ner_by_cond, f"{exp_id} (test)",
                                PATHS["figures_root"] / f"{exp_id}_ner_test.png")
    figures.plot_path_metrics(summary, f"{exp_id} (test)",
                              PATHS["figures_root"] / f"{exp_id}_path_metrics.png")
    print(f"[done] {exp_id}: summary -> {metrics_dir / 'summary.json'}")
    return out


def run_dual_experiment(dataset_key: str, seeds: List[int], device_arg: str = "auto",
                        debug: bool = False, skip_existing: bool = False,
                        conditions: Optional[List[str]] = None, sweep: bool = False,
                        ner_lambda: Optional[float] = None) -> None:
    ensure_dist()
    specs = resolve_conditions(conditions, sweep=sweep, ner_lambda=ner_lambda)
    for condition, spec in specs:
        for seed in seeds:
            run_condition_3(condition, spec, dataset_key, seed, device_arg=device_arg,
                            debug=debug, skip_existing=skip_existing)
    if dist_rank() == 0:
        summarize_exp3()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--experiment", default=EXP_3["experiment"])
    args = parser.parse_args()
    if args.summarize_only:
        summarize_exp3(args.experiment)


if __name__ == "__main__":
    main()
