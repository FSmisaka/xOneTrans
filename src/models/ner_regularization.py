"""Differentiable spectral quantities for NER regularization (SPEC3 section 4).

The evaluation-side NER (`src.experiments.diagnostics.normalized_effective_rank`)
column-centers the valid S-token matrix, takes the singular values, normalizes
them into a probability vector p_i = sigma_i / sum_j sigma_j, computes
ER = exp(-sum_i p_i log p_i) and NER = ER / D. This module mirrors that
definition differentiably for training (and for the per-batch observed-NER log).

The Gram path (`X^T X` + `eigvalsh`) is used when the token matrix is tall and
well conditioned; rank-deficient / wide matrices fall back to `svdvals` so that
zero singular values are represented exactly as in the evaluation code.
"""

import math
from typing import Dict, Optional

import torch


def _spectral_values(x_centered: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    n, d = x_centered.shape
    if backend in ("auto", "gram") and n >= d:
        gram = x_centered.transpose(0, 1) @ x_centered
        gram = 0.5 * (gram + gram.transpose(0, 1))
        eig = torch.linalg.eigvalsh(gram)
        max_eig = float(eig[-1].detach())
        min_eig = float(eig[0].detach())
        if max_eig > 0.0 and min_eig > 1e-8 * max_eig:
            return torch.sqrt(eig.clamp_min(0.0))
    return torch.linalg.svdvals(x_centered)


def differentiable_ner_stats(states: torch.Tensor, mask: Optional[torch.Tensor] = None,
                             epsilon: float = 1e-8, backend: str = "auto") -> Dict:
    """Spectral entropy / NER of the valid sequence tokens in `states`.

    `states` is [B, L, D]; `mask` is [B, L] and padding positions are dropped
    before any spectral statistic. Returns tensors (`ner`, `h_norm`, `entropy`)
    and integer token/spectrum counts. `h_norm = H / log(min(N, D))` is the
    stabilized training scale; `ner` matches the evaluation definition.
    """
    x = states.float()
    if mask is not None:
        x = x[mask.bool()]
    n, d = x.shape
    if n < 2:
        zero = x.sum() * 0.0
        return {"ner": zero, "h_norm": zero, "entropy": zero,
                "n_tokens": int(n), "n_spectral": int(min(n, d))}
    x_centered = x - x.mean(dim=0, keepdim=True)
    sigma = _spectral_values(x_centered, backend)
    p = sigma / sigma.sum().clamp_min(epsilon)
    entropy = -(p * torch.log(p.clamp_min(epsilon))).sum()
    rank = min(n, d)
    h_norm = entropy / math.log(rank) if rank > 1 else entropy * 0.0
    ner = torch.exp(entropy) / d
    return {"ner": ner, "h_norm": h_norm, "entropy": entropy,
            "n_tokens": int(n), "n_spectral": int(rank)}
