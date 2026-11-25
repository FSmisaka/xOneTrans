"""Shared-OneTrans continuous spectral mixture (SPEC4).

One frozen prefix + one shared last OneTrans block + one shared CTR tower.
For each alpha node only the S-stream entering the last block is transformed by
a fixed train-reference spectral transform. Expert modes run the shared final
block once per alpha; the representation mode fuses transformed S streams and
runs the final block once.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.onetrans_diagnostic import MLPBlock, OneTransDiagnostic
from src.models.spectral_transform import apply_spectral_transform


def logit_from_prob(prob: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = prob.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def simpson_weights(scores: torch.Tensor, quad_weights: torch.Tensor) -> torch.Tensor:
    """w_bk = softmax_k(s_bk + log(quad_weight_k)); zeros -> Simpson weights."""
    q = quad_weights.to(device=scores.device, dtype=scores.dtype).clamp_min(1e-30)
    return torch.softmax(scores + torch.log(q), dim=-1)


def probability_mix(probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (weights * probs).sum(dim=-1)


def _zero_init_last(mlp: MLPBlock) -> None:
    last = mlp.net[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class ContinuousSpectralOneTrans(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], model_cfg: Dict,
                 condition_cfg: Dict, spectral_cfg: Dict, gate_hidden_dim: int = 128,
                 gate_dropout: float = 0.0):
        super().__init__()
        self.kind = condition_cfg["kind"]
        self.fusion = condition_cfg.get("fusion")
        self.fixed_alpha = float(condition_cfg.get("alpha", 1.0))
        self.token_dim = int(model_cfg["token_dim"])
        self.num_layers = int(model_cfg["num_layers"])
        self.num_ns_token = int(model_cfg["num_ns_token"])
        self.readout_dim = self.token_dim * self.num_ns_token
        self.base = OneTransDiagnostic(vocab_sizes, model_cfg)
        self.alpha_nodes_all = torch.tensor(spectral_cfg["alpha_nodes"], dtype=torch.float32)
        self.quad_weights_all = torch.tensor(spectral_cfg["quadrature_weights"],
                                             dtype=torch.float32)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_dropout = float(gate_dropout)
        self.register_buffer("mu_ref", torch.zeros(self.token_dim))
        self.register_buffer("T_alphas", torch.zeros(0, self.token_dim, self.token_dim))
        self.register_buffer("sigma_alphas", torch.zeros(0, self.token_dim))
        self.register_buffer("active_alphas", torch.zeros(0))
        self.register_buffer("active_quad_weights", torch.zeros(0))
        self.register_buffer("cont_V", torch.zeros(self.token_dim, self.token_dim))
        self.register_buffer("cont_r", torch.zeros(0, self.token_dim))
        self.register_buffer("cont_alphas", torch.zeros(0))
        self.register_buffer("cont_quad_weights", torch.zeros(0))
        self.continuous_cfg = spectral_cfg.get("continuous")
        self.basis_metadata: Dict = {}
        self.baseline_checkpoint: Optional[str] = None

        if self.fusion == "global":
            if self.kind != "expert_mix":
                raise ValueError("global fusion is only defined for expert_mix")
            self.global_gate = MLPBlock(2, [self.gate_hidden_dim],
                                        hidden_activations=model_cfg["dnn_activations"],
                                        output_dim=1, dropout_rates=self.gate_dropout)
            _zero_init_last(self.global_gate)
        if self.fusion == "contextual":
            context_dim = self.token_dim + self.readout_dim
            self.context_gate = MLPBlock(context_dim + 2, [self.gate_hidden_dim],
                                         hidden_activations=model_cfg["dnn_activations"],
                                         output_dim=1, dropout_rates=self.gate_dropout)
            _zero_init_last(self.context_gate)
        if self.kind in ("representation_mix", "continuous_representation_mix") \
                and self.fusion != "contextual":
            raise ValueError(f"{self.kind} requires contextual fusion")
        if self.kind == "single" and self.fusion is not None:
            raise ValueError("single-alpha conditions do not use a fusion gate")
        if self.kind == "continuous_representation_mix":
            if not self.continuous_cfg:
                raise ValueError("continuous_representation_mix requires a spectral.continuous config")
            q = int(self.continuous_cfg["num_nodes"])
            alpha_min = float(self.continuous_cfg["alpha_min"])
            alpha_max = float(self.continuous_cfg["alpha_max"])
            if q < 2 or alpha_min <= 0 or alpha_max <= alpha_min:
                raise ValueError("invalid continuous alpha integration config")
            t_min, t_max = math.log(alpha_min), math.log(alpha_max)
            x, w = np.polynomial.legendre.leggauss(q)
            t = 0.5 * (t_max - t_min) * x + 0.5 * (t_max + t_min)
            weights = w * 0.5 * (t_max - t_min)
            weights = weights / weights.sum()
            self.cont_alphas = torch.tensor(np.exp(t), dtype=torch.float32)
            self.cont_quad_weights = torch.tensor(weights, dtype=torch.float32)

    def load_baseline(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        self.base.load_state_dict(state, strict=True)
        self.baseline_checkpoint = str(checkpoint_path)

    def set_spectral_basis(self, basis: Dict, transforms: List[Dict],
                           alpha_nodes: List[float], quad_weights: List[float]) -> None:
        self.mu_ref = basis["mu_ref"].float().to(self.mu_ref.device)
        if self.kind == "continuous_representation_mix":
            sigma = basis["sigma"].float().to(self.mu_ref.device)
            v = basis["V"].float().to(self.mu_ref.device)
            tol = float(basis.get("metadata", {}).get("rank_relative_tolerance", 1e-6))
            if float(sigma.max()) <= 0.0:
                raise ValueError("cannot build continuous basis from a zero spectrum")
            significant = sigma > sigma.max() * tol
            if not significant.any():
                raise ValueError("continuous basis has no significant spectral directions")
            log_sigma = torch.log(sigma[significant])
            alphas = self.cont_alphas.to(device=self.mu_ref.device, dtype=torch.float32)
            scores = alphas.unsqueeze(1) * log_sigma.unsqueeze(0)
            probabilities = torch.softmax(scores, dim=1)
            sigma_alpha = sigma.sum() * probabilities
            ratios = torch.zeros(alphas.numel(), sigma.numel(),
                                 device=self.mu_ref.device, dtype=torch.float32)
            ratios[:, significant] = sigma_alpha / sigma[significant]
            self.cont_V = v
            self.cont_r = ratios
            self.active_alphas = alphas
            self.active_quad_weights = self.cont_quad_weights.to(
                device=self.mu_ref.device, dtype=torch.float32)
            self.basis_metadata = dict(basis.get("metadata", {}))
            return
        if self.kind == "single":
            idx = int(min(range(len(alpha_nodes)),
                          key=lambda i: abs(float(alpha_nodes[i]) - self.fixed_alpha)))
            active_alphas = [float(alpha_nodes[idx])]
            active_quad = [1.0]
            active_transforms = [transforms[idx]]
            active_sigma = [transforms[idx]["sigma_alpha"]]
        else:
            active_alphas = [float(a) for a in alpha_nodes]
            active_quad = [float(q) for q in quad_weights]
            active_transforms = list(transforms)
            active_sigma = [tr["sigma_alpha"] for tr in transforms]
        self.T_alphas = torch.stack([tr["T"].float() for tr in active_transforms]).to(self.mu_ref.device)
        self.sigma_alphas = torch.stack([s.float() for s in active_sigma]).to(self.mu_ref.device)
        self.active_alphas = torch.tensor(active_alphas, dtype=torch.float32, device=self.mu_ref.device)
        q = torch.tensor(active_quad, dtype=torch.float32, device=self.mu_ref.device)
        self.active_quad_weights = q / q.sum().clamp_min(1e-30)
        self.basis_metadata = dict(basis.get("metadata", {}))

    def freeze_prefix(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = False
        last = self.num_layers - 1
        trainable_modules = [
            self.base.backbone.norm1_layers[last], self.base.backbone.norm2_layers[last],
            self.base.backbone.mha_layers[last], self.base.backbone.ffn_layers[last],
            self.base.tower,
        ]
        for module in trainable_modules:
            for p in module.parameters():
                p.requires_grad = True
        self.base.residual_gate = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.base.item_embedding.eval()
            self.base.action_embedding.eval()
            self.base.field_embeddings.eval()
            self.base.item_token_proj.eval()
            self.base.unified_tokenizer.eval()
            for i in range(self.num_layers - 1):
                self.base.backbone.norm1_layers[i].eval()
                self.base.backbone.norm2_layers[i].eval()
                self.base.backbone.mha_layers[i].eval()
                self.base.backbone.ffn_layers[i].eval()
        return self

    def param_groups(self, dense_lr: float, sparse_lr: float):
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": dense_lr}] if params else []

    def _context(self, s_pre: torch.Tensor, ns_pre: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        s_pool = (s_pre * mask.unsqueeze(-1)).sum(dim=1) / denom
        return torch.cat([s_pool, ns_pre.flatten(start_dim=1)], dim=-1)

    def _gate_scores(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        t = torch.log(self.active_alphas)
        if self.fusion == "uniform":
            return torch.zeros(batch_size, self.active_alphas.numel(),
                               device=context.device, dtype=context.dtype)
        t_feat = torch.stack([t, t * t], dim=-1)
        if self.fusion == "global":
            return self.global_gate(t_feat).squeeze(-1).unsqueeze(0).expand(batch_size, -1)
        if self.fusion == "contextual":
            k = self.active_alphas.numel()
            ctx = context.unsqueeze(1).expand(batch_size, k, context.shape[-1])
            t_expand = t_feat.unsqueeze(0).expand(batch_size, k, 2)
            features = torch.cat([ctx, t_expand], dim=-1).reshape(batch_size * k, -1)
            return self.context_gate(features).view(batch_size, k)
        raise ValueError(f"unknown fusion mode: {self.fusion}")

    def _mixture_weights(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.fusion == "uniform":
            return self.active_quad_weights.unsqueeze(0).expand(batch_size, -1)
        scores = self._gate_scores(context, batch_size)
        return simpson_weights(scores, self.active_quad_weights)

    def _transformed_s(self, s_pre: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        return [
            apply_spectral_transform(s_pre, mask, self.mu_ref, self.T_alphas[k],
                                     alpha=float(self.active_alphas[k]))
            for k in range(self.active_alphas.numel())
        ]

    def _final(self, s_tokens: torch.Tensor, ns_pre: torch.Tensor,
               mask: torch.Tensor) -> Dict:
        return self.base.final_layer(s_tokens, ns_pre, mask,
                                     layer_idx=self.num_layers - 1)

    def forward(self, batch: Dict[str, torch.Tensor], field_mask: Optional[torch.Tensor] = None,
                return_states: bool = False):
        with torch.no_grad():
            s_pre, ns_pre, mask = self.base.encode_prefix(batch, field_mask)
        batch_size = s_pre.shape[0]

        if self.kind == "continuous_representation_mix":
            context = self._context(s_pre, ns_pre, mask)
            weights = self._mixture_weights(context, batch_size)
            r_bar = torch.matmul(weights, self.cont_r)
            centered = s_pre - self.mu_ref
            z = torch.matmul(centered, self.cont_V) * r_bar.unsqueeze(1)
            s_mix = self.mu_ref + torch.matmul(z, self.cont_V.transpose(0, 1))
            s_mix = s_mix * mask.unsqueeze(-1).to(s_mix.dtype)
            final = self._final(s_mix, ns_pre, mask)
            logit = final["logit"]
            prob = torch.sigmoid(logit)
            out = {
                "logit": logit, "pred": prob, "mix_logit": logit, "mix_prob": prob,
                "weights": weights, "alpha_nodes": self.active_alphas,
                "quad_weights": self.active_quad_weights, "fusion": self.fusion,
                "hist_mask": mask, "final_mask": final["hist_mask"],
            }
            if return_states:
                out["s_pre"] = s_pre
                out["s_mix"] = s_mix
                out["s_final_mix"] = final["s_final"]
            return out

        s_alphas = self._transformed_s(s_pre, mask)

        if self.kind == "single":
            final = self._final(s_alphas[0], ns_pre, mask)
            logit = final["logit"]
            prob = torch.sigmoid(logit)
            out = {
                "logit": logit, "pred": prob, "mix_logit": logit, "mix_prob": prob,
                "logits_by_alpha": logit.unsqueeze(1),
                "weights": torch.ones(batch_size, 1, device=logit.device, dtype=logit.dtype),
                "alpha_nodes": self.active_alphas, "quad_weights": self.active_quad_weights,
                "fusion": "single", "hist_mask": mask, "final_mask": final["hist_mask"],
            }
            if return_states:
                out["s_pre"] = s_pre
                out["s_alpha"] = s_alphas[0].unsqueeze(1)
                out["s_final"] = final["s_final"].unsqueeze(1)
            return out

        context = self._context(s_pre, ns_pre, mask)
        weights = self._mixture_weights(context, batch_size)

        if self.kind == "expert_mix":
            logits, s_finals = [], []
            final_mask = None
            for s_alpha in s_alphas:
                final = self._final(s_alpha, ns_pre, mask)
                logits.append(final["logit"])
                final_mask = final["hist_mask"]
                if return_states:
                    s_finals.append(final["s_final"])
            logits = torch.stack(logits, dim=1)
            probs = torch.sigmoid(logits)
            mix_prob = probability_mix(probs, weights)
            mix_logit = logit_from_prob(mix_prob)
            out = {
                "logit": mix_logit, "pred": mix_prob,
                "mix_logit": mix_logit, "mix_prob": mix_prob,
                "logits_by_alpha": logits, "weights": weights,
                "alpha_nodes": self.active_alphas, "quad_weights": self.active_quad_weights,
                "fusion": self.fusion, "hist_mask": mask, "final_mask": final_mask,
            }
            if return_states:
                out["s_pre"] = s_pre
                out["s_alpha"] = torch.stack(s_alphas, dim=1)
                out["s_final"] = torch.stack(s_finals, dim=1)
            return out

        if self.kind == "representation_mix":
            s_alpha_stack = torch.stack(s_alphas, dim=1)
            s_mix = torch.einsum("bk,bkld->bld", weights, s_alpha_stack)
            final = self._final(s_mix, ns_pre, mask)
            logit = final["logit"]
            prob = torch.sigmoid(logit)
            out = {
                "logit": logit, "pred": prob, "mix_logit": logit, "mix_prob": prob,
                "weights": weights, "alpha_nodes": self.active_alphas,
                "quad_weights": self.active_quad_weights, "fusion": self.fusion,
                "hist_mask": mask, "final_mask": final["hist_mask"],
            }
            if return_states:
                out["s_pre"] = s_pre
                out["s_alpha"] = s_alpha_stack
                out["s_mix"] = s_mix
                out["s_final_mix"] = final["s_final"]
            return out

        raise ValueError(f"unknown kind: {self.kind}")
