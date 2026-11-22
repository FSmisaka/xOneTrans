"""Shared training/evaluation entry point for all diagnostic experiments."""

import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS
from config.dataset import CONFIG as DATA_CFG
from config.onetrans import CONFIG as MODEL_CFG
from config.sequence_probe import CONFIG as PROBE_CFG
from src.data.dataset import KuaiRandData, KuaiRandSequenceDataset, QualityTransform, collate_batch, VARIABLE_FIELDS
from src.models.onetrans_diagnostic import OneTransDiagnostic
from src.experiments.diagnostics import (collect_states, compute_layer_ner, fit_sequence_probe,
                                         select_diagnostic_indices, select_ner_token_pairs)
from src.utils.logging import setup_logger
from src.utils.metrics import auc_score, logloss_score
from src.utils.seed import set_seed

_DATA_CACHE: Dict[str, KuaiRandData] = {}


def get_data(dataset_key: str) -> KuaiRandData:
    if dataset_key not in _DATA_CACHE:
        _DATA_CACHE[dataset_key] = KuaiRandData(dataset_key)
    return _DATA_CACHE[dataset_key]


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_field_mask(enabled_fields) -> torch.Tensor:
    mask = [1.0 if f in set(enabled_fields) else 0.0 for f in VARIABLE_FIELDS]
    return torch.tensor(mask + [1.0])  # last slot: target item id, always on


class MaskedOneTrans(nn.Module):
    """Applies a fixed variable-field mask to every forward pass."""

    def __init__(self, model: OneTransDiagnostic, field_mask: Optional[torch.Tensor]):
        super().__init__()
        self.model = model
        self.field_mask = field_mask

    def forward(self, batch, **kw):
        return self.model(batch, field_mask=self.field_mask, **kw)

    def param_groups(self, dense_lr: float, sparse_lr: float):
        return self.model.param_groups(dense_lr, sparse_lr)

    @property
    def token_dim(self):
        return self.model.token_dim


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        preds.append(out["pred"].float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return {"y_pred": np.concatenate(preds), "y_true": np.concatenate(labels)}


def train_model(model, datasets, seed, experiment, logger, device, ddp=False, epochs=None,
                accum_steps: int = 1, eval_batch_size: int = None):
    cfg = MODEL_CFG
    n_epochs = epochs or cfg["epochs"]
    micro_batch = max(1, cfg["batch_size"] // accum_steps)
    if logger is not None and accum_steps > 1:
        logger.info(f"gradient accumulation: micro_batch={micro_batch} x {accum_steps}")
    train_sampler = None
    shuffle = True
    if ddp:
        train_sampler = torch.utils.data.DistributedSampler(datasets["train"], shuffle=True, seed=seed)
        shuffle = False
    train_loader = DataLoader(datasets["train"], batch_size=micro_batch, shuffle=shuffle,
                              sampler=train_sampler, collate_fn=collate_batch,
                              num_workers=cfg["num_workers"], drop_last=False)
    valid_loader = DataLoader(datasets["valid"], batch_size=eval_batch_size or cfg["eval_batch_size"],
                              shuffle=False, collate_fn=collate_batch, num_workers=cfg["num_workers"])
    optimizer = torch.optim.Adam(model.param_groups(cfg["dense_learning_rate"],
                                                    cfg["sparse_learning_rate"]),
                                 weight_decay=cfg["weight_decay"])
    amp_dtype = {"none": None, "bf16": torch.bfloat16}.get(cfg["amp_dtype"])
    best_auc, best_state, best_epoch = -1.0, None, -1
    patience_left = cfg["early_stop_patience"]
    train_time, epochs_run, last_loss = 0.0, 0, float("nan")

    for epoch in range(n_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(batch)
                    loss = F.binary_cross_entropy_with_logits(out["logit"].float(), batch["label"])
            else:
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(out["logit"], batch["label"])
            (loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last_loss = loss.item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_time += time.time() - t0
        epochs_run = epoch + 1

        auc = None
        if logger is not None:
            res = evaluate(model, valid_loader, device)
            auc = auc_score(res["y_true"], res["y_pred"])
            logger.info(f"epoch {epoch + 1}: valid_auc={auc:.6f} train_loss={last_loss:.6f}")
        stop = torch.zeros(1, device=device)
        if auc is not None:
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch + 1
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                patience_left = cfg["early_stop_patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    stop.fill_(1.0)
        if ddp:
            torch.distributed.broadcast(stop, src=0)
        if stop.item() > 0:
            if logger is not None:
                logger.info(f"early stop at epoch {epoch + 1} (best epoch {best_epoch}, auc={best_auc:.6f})")
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    throughput = len(datasets["train"]) * epochs_run / max(train_time, 1e-9)
    return model, {"best_valid_auc": best_auc, "best_epoch": best_epoch, "epochs_run": epochs_run,
                   "throughput_samples_per_sec": throughput}


def accum_steps_for(max_len: int, batch_size: int, device: torch.device) -> int:
    """Gradient-accumulation steps keeping the effective batch fixed. CUDA SDPA
    uses flash kernels (no score materialization); other backends need smaller
    micro-batches for long sequences."""
    base = max(1, int(np.ceil(max_len / 100)))
    if device.type == "cuda":
        return base
    max_micro = max(32, int(64 * (500 / max(max_len, 1)) ** 2))
    return max(base, int(np.ceil(batch_size / max_micro)))


def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def run_condition(experiment: str, condition: str, dataset_key: str, seed: int,
                  enabled_fields=None, transform: Optional[QualityTransform] = None,
                  max_history_len: Optional[int] = None, min_history: Optional[int] = None,
                  device_arg: str = "auto", debug: bool = False) -> Optional[Dict]:
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    if ddp:
        torch.distributed.init_process_group("nccl")
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        device = resolve_device(device_arg)
    logger = setup_logger(f"{experiment}_{seed}", PATHS["logs_root"] / experiment / f"{seed}.log") if rank == 0 else None
    if logger:
        logger.info(f"experiment={experiment} condition={condition} dataset={dataset_key} "
                    f"seed={seed} device={device} ddp={ddp}")

    set_seed(seed)
    data = get_data(dataset_key)
    max_len = max_history_len or MODEL_CFG["max_history_len"]
    dbg = MODEL_CFG["debug"] if debug else None
    datasets = {
        split: KuaiRandSequenceDataset(
            data, split, max_len, transform=transform, min_history=min_history,
            max_rows=(dbg or {}).get(f"{split}_samples"))
        for split in ("train", "valid", "test")
    }
    if logger:
        logger.info("samples: " + ", ".join(f"{s}={len(d)}" for s, d in datasets.items()))

    base = OneTransDiagnostic(data.vocab_sizes(), MODEL_CFG).to(device)
    field_mask = build_field_mask(enabled_fields).to(device) if enabled_fields is not None else None
    model = MaskedOneTrans(base, field_mask)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    accum_steps = accum_steps_for(max_len, MODEL_CFG["batch_size"], device)
    eval_bs = max(256, MODEL_CFG["eval_batch_size"] * MODEL_CFG["max_history_len"] // max_len)
    model, train_info = train_model(model, datasets, seed, experiment, logger, device, ddp=ddp,
                                    epochs=(dbg or {}).get("epochs"), accum_steps=accum_steps,
                                    eval_batch_size=eval_bs)
    if logger:
        logger.info(f"training done: {train_info}")

    metrics = None
    if rank == 0:
        core = model.module if ddp else model
        valid_loader = DataLoader(datasets["valid"], batch_size=eval_bs, shuffle=False,
                                  collate_fn=collate_batch, num_workers=MODEL_CFG["num_workers"])
        test_loader = DataLoader(datasets["test"], batch_size=eval_bs, shuffle=False,
                                 collate_fn=collate_batch, num_workers=MODEL_CFG["num_workers"])
        res_valid = evaluate(core, valid_loader, device)
        res_test = evaluate(core, test_loader, device)
        full_metrics = {
            "full_valid_auc": auc_score(res_valid["y_true"], res_valid["y_pred"]),
            "full_valid_logloss": logloss_score(res_valid["y_true"], res_valid["y_pred"]),
            "full_test_auc": auc_score(res_test["y_true"], res_test["y_pred"]),
            "full_test_logloss": logloss_score(res_test["y_true"], res_test["y_pred"]),
        }

        diag_idx = select_diagnostic_indices(datasets["valid"], PROBE_CFG["diagnostic_samples"],
                                             PROBE_CFG["diagnostic_seed"])
        lengths = np.minimum(datasets["valid"].position[diag_idx], max_len)
        rank_arr, pos_arr = select_ner_token_pairs(lengths, PROBE_CFG["ner_max_tokens"],
                                                   PROBE_CFG["ner_seed"])
        valid_states = collect_states(core, datasets["valid"], diag_idx, rank_arr, pos_arr,
                                      eval_bs, device, pool_last_layer=True)
        ner = compute_layer_ner(valid_states["tokens"])

        probe_rng = np.random.default_rng(PROBE_CFG["diagnostic_seed"])
        probe_train_idx = np.sort(probe_rng.permutation(len(datasets["train"]))[: PROBE_CFG["probe_train_max"]])
        probe_test_idx = np.arange(min(PROBE_CFG["probe_eval_max"], len(datasets["test"])))
        empty_r = np.zeros(0, dtype=np.int64)
        empty_p = np.zeros(0, dtype=np.int64)
        train_states = collect_states(core, datasets["train"], probe_train_idx, empty_r, empty_p,
                                      eval_bs, device, pool_last_layer=True)
        test_states = collect_states(core, datasets["test"], probe_test_idx, empty_r, empty_p,
                                     eval_bs, device, pool_last_layer=True)
        probe_metrics = fit_sequence_probe(
            train_states["pooled"].numpy(), train_states["labels"].numpy(),
            valid_states["pooled"].numpy(), valid_states["labels"].numpy(),
            test_states["pooled"].numpy(), test_states["labels"].numpy(),
        )

        ckpt_dir = PATHS["checkpoints_root"] / experiment / str(seed)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best.pt"
        torch.save({"model_state": base.state_dict(), "field_mask": field_mask,
                    "condition": condition, "seed": seed}, ckpt_path)

        meta_path = PATHS["metadata_root"] / f"processed_{dataset_key}.json"
        processed_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metrics = {
            "experiment": experiment,
            "condition": condition,
            "dataset": dataset_key,
            "dataset_version": DATA_CFG["dataset_names"].get(dataset_key, dataset_key),
            "processed_meta": processed_meta,
            "unirank_reference": PATHS["unirank_reference"],
            "config_snapshot": {"onetrans": MODEL_CFG, "probe": PROBE_CFG,
                                "max_history_len": max_len, "min_history": min_history,
                                "enabled_fields": enabled_fields,
                                "transformed": transform is not None,
                                "debug": debug},
            "seed": seed,
            "sample_counts": {s: len(d) for s, d in datasets.items()},
            "diagnostic_samples": int(len(diag_idx)),
            "ner_tokens": int(len(rank_arr)),
            "training": {**train_info, "accum_steps": accum_steps, "micro_batch_size": max(1, MODEL_CFG["batch_size"] // accum_steps)},
            "ner": ner,
            **probe_metrics,
            **full_metrics,
            "training_cost": {
                "n_params": sum(p.numel() for p in base.parameters()),
                "throughput_samples_per_sec": train_info["throughput_samples_per_sec"],
                "peak_memory_mb": peak_memory_mb(device),
                "device": str(device),
                "ddp_world_size": int(os.environ.get("WORLD_SIZE", "1")),
            },
            "best_checkpoint": str(ckpt_path),
        }
        metrics_dir = PATHS["metrics_root"] / experiment
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / f"{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"ner={ner}")
        logger.info(f"metrics: probe={probe_metrics} full={full_metrics}")

    if ddp:
        torch.distributed.destroy_process_group()
    return metrics
