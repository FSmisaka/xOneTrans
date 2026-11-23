"""Dual independent OneTrans branches with three CTR heads (SPEC3 section 3).

The high- and low-NER branches are two complete `OneTransDiagnostic` instances
with separate parameters. The second branch is a deep copy of the first one, so
both start from the same initialization and then update independently. Each
branch keeps the original OneTrans CTR tower as its own head; the mixed head is
a light MLP over the concatenation of the two CTR readouts. The mixed loss is
not detached, so it trains both branches end to end.
"""

import copy
from typing import Dict, Optional

import torch
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.onetrans_diagnostic import MLPBlock, OneTransDiagnostic


class DualBranchOneTrans(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], model_cfg: Dict):
        super().__init__()
        cfg = dict(model_cfg)
        if cfg.get("use_residual_gate", False):
            raise ValueError("SPEC3 dual-branch runs do not use the residual gate")
        self.token_dim = cfg["token_dim"]
        self.num_ns_token = cfg["num_ns_token"]
        self.readout_dim = self.token_dim * self.num_ns_token

        self.high_branch = OneTransDiagnostic(vocab_sizes, cfg)
        self.low_branch = copy.deepcopy(self.high_branch)

        mix_cfg = cfg.get("mix_head") or {}
        hidden_dim = mix_cfg.get("hidden_dim") or self.readout_dim
        dropout = mix_cfg.get("dropout")
        if dropout is None:
            dropout = cfg.get("net_dropout", 0.0)
        self.mix_head = MLPBlock(
            2 * self.readout_dim, [hidden_dim],
            hidden_activations=cfg["dnn_activations"], output_dim=1,
            layer_norm=mix_cfg.get("layer_norm", False), dropout_rates=dropout,
        )

    def param_groups(self, dense_lr: float, sparse_lr: float):
        sparse_ids = set()
        for branch in (self.high_branch, self.low_branch):
            embeddings = [branch.item_embedding, branch.action_embedding,
                          *branch.field_embeddings.values()]
            for emb in embeddings:
                sparse_ids.update(id(p) for p in emb.parameters())
        dense, sparse = [], []
        for p in self.parameters():
            (sparse if id(p) in sparse_ids else dense).append(p)
        return [{"params": dense, "lr": dense_lr}, {"params": sparse, "lr": sparse_lr}]

    def forward(self, batch: Dict[str, torch.Tensor], field_mask: Optional[torch.Tensor] = None,
                return_sequence_states: bool = False):
        kwargs = dict(field_mask=field_mask, return_readout=True, return_last_state=True,
                      return_sequence_states=return_sequence_states)
        high = self.high_branch(batch, **kwargs)
        low = self.low_branch(batch, **kwargs)

        z_high, z_low = high["readout"], low["readout"]
        logit_high, logit_low = high["logit"], low["logit"]
        logit_mix = self.mix_head(torch.cat([z_high, z_low], dim=-1)).squeeze(-1)
        logit_avg = 0.5 * (logit_high + logit_low)

        out = {
            "logit_high": logit_high, "logit_low": logit_low, "logit_mix": logit_mix,
            "logit_avg": logit_avg,
            "pred_high": torch.sigmoid(logit_high), "pred_low": torch.sigmoid(logit_low),
            "pred_mix": torch.sigmoid(logit_mix), "pred_avg": torch.sigmoid(logit_avg),
            "logit": logit_mix, "pred": torch.sigmoid(logit_mix),
            "readout_high": z_high, "readout_low": z_low,
            "last_state_high": high["last_state"], "last_state_low": low["last_state"],
            "hist_mask": high["hist_mask"],
        }
        if return_sequence_states:
            out["states_high"] = high["states"]
            out["states_low"] = low["states"]
        return out
