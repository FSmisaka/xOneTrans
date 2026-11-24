"""Fixed train-reference spectral transforms for Exp4 (SPEC4 section 2).

The reference basis is fitted once from valid S-token states of the *train*
split. It is then fixed for the whole run. For an alpha node, the singular
values are reweighted in log domain while the reference nuclear norm
sum(sigma) is preserved; alpha=1 is an exact identity.

Both the analytic NER and the empirical NER use the project definition:
column-centering, singular values sigma_i, p_i = sigma_i / sum_j sigma_j,
ER = exp(-sum_i p_i log p_i), NER = ER / token_dim.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _validate_reference(reference_states: torch.Tensor,
                        reference_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if reference_states.dim() == 2:
        x = reference_states
    elif reference_states.dim() == 3:
        if reference_mask is None:
            raise ValueError("reference_mask is required when reference_states is [B, L, D]")
        x = reference_states[reference_mask.bool()]
    else:
        raise ValueError("reference_states must be [N, D] or [B, L, D]")
    if x.shape[0] < 2:
        raise ValueError(f"need at least 2 calibration tokens, got {x.shape[0]}")
    if not torch.isfinite(x).all():
        raise ValueError("reference_states contains NaN or Inf")
    return x


def fit_spectral_basis(reference_states: torch.Tensor,
                       reference_mask: Optional[torch.Tensor] = None,
                       rank_relative_tolerance: float = 1e-6,
                       svd_dtype: str = "float64",
                       metadata: Optional[Dict] = None) -> Dict:
    """Fit mu_ref, V and sigma from train-only reference S-token states.

    A full right singular basis V in R^{D x D} is returned. When N < D the
    remaining singular directions are explicitly zero. An empty, all-zero or
    non-finite reference matrix raises instead of being silently regularized.
    """
    x = _validate_reference(reference_states, reference_mask)
    if rank_relative_tolerance <= 0:
        raise ValueError("rank_relative_tolerance must be positive")
    dtype = torch.float64 if str(svd_dtype).lower() == "float64" else torch.float32
    x = x.to(dtype)
    mu_ref = x.mean(dim=0)
    x_centered = x - mu_ref
    full_matrices = x.shape[0] < x.shape[1]
    _, singular, vh = torch.linalg.svd(x_centered, full_matrices=full_matrices)
    v = vh.transpose(0, 1).contiguous()
    token_dim = int(x.shape[1])
    sigma = torch.zeros(token_dim, dtype=dtype, device=x.device)
    sigma[: singular.shape[0]] = singular
    if not torch.isfinite(sigma).all():
        raise ValueError("singular values contain NaN or Inf")
    if float(sigma.max()) <= 0.0:
        raise ValueError("reference matrix has a zero spectrum")
    significant = sigma > sigma.max() * rank_relative_tolerance
    meta = {
        "calibration_split": (metadata or {}).get("calibration_split", "train"),
        "n_tokens": int(x.shape[0]),
        "token_dim": token_dim,
        "n_singular_values": int(singular.shape[0]),
        "n_significant": int(significant.sum()),
        "rank_relative_tolerance": float(rank_relative_tolerance),
        "svd_dtype": str(svd_dtype),
    }
    if metadata:
        meta.update({k: v for k, v in metadata.items() if k != "calibration_split"})
        meta["calibration_split"] = metadata.get("calibration_split", "train")
    return {"mu_ref": mu_ref.cpu(), "V": v.cpu(), "sigma": sigma.cpu(), "metadata": meta}


def _sigma_alpha_from_spectrum(sigma: torch.Tensor, alpha: float,
                               rank_relative_tolerance: float) -> Tuple[torch.Tensor, torch.Tensor]:
    sigma = sigma.to(torch.float64) if sigma.dtype == torch.float64 else sigma.float()
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if alpha == 1.0:
        return sigma.clone(), torch.ones_like(sigma, dtype=torch.bool)
    sigma_sum = sigma.sum()
    sigma_max = sigma.max()
    if float(sigma_max) <= 0.0 or float(sigma_sum) <= 0.0:
        raise ValueError("cannot transform a zero spectrum")
    significant = sigma > sigma_max * rank_relative_tolerance
    log_sigma = torch.log(sigma[significant])
    scores = float(alpha) * log_sigma
    probabilities = F.softmax(scores, dim=0)
    sigma_alpha = torch.zeros_like(sigma)
    sigma_alpha[significant] = sigma_sum * probabilities
    return sigma_alpha, significant


def analytic_ner_from_spectrum(sigma: torch.Tensor, alpha: float = 1.0,
                               token_dim: Optional[int] = None,
                               rank_relative_tolerance: float = 1e-6,
                               epsilon: float = 1e-30) -> float:
    sigma_alpha, _ = _sigma_alpha_from_spectrum(sigma, alpha, rank_relative_tolerance)
    total = sigma_alpha.sum()
    if float(total) <= 0.0:
        return 0.0
    p = sigma_alpha / total
    entropy = -(p * torch.log(p.clamp_min(epsilon))).sum()
    er = torch.exp(entropy)
    dim = int(token_dim if token_dim is not None else sigma_alpha.numel())
    return float(er / dim)


def build_spectral_transform(sigma: torch.Tensor, v: torch.Tensor, alpha: float,
                             token_dim: Optional[int] = None,
                             rank_relative_tolerance: float = 1e-6,
                             epsilon: float = 1e-30) -> Dict:
    """Build T_alpha = V diag(r_i) V^T and the transformed reference spectrum."""
    if sigma.dim() != 1 or v.dim() != 2 or v.shape[0] != v.shape[1]:
        raise ValueError("sigma must be [D] and v must be [D, D]")
    if sigma.numel() != v.shape[0]:
        raise ValueError("sigma and v dimensions do not match")
    if not torch.isfinite(sigma).all() or not torch.isfinite(v).all():
        raise ValueError("spectral basis contains NaN or Inf")
    dtype = torch.float64 if sigma.dtype == torch.float64 else sigma.float()
    sigma = sigma.to(dtype)
    v = v.to(dtype)
    token_dim = int(token_dim if token_dim is not None else sigma.numel())
    significant = sigma > sigma.max() * rank_relative_tolerance if float(sigma.max()) > 0 else \
        torch.zeros_like(sigma, dtype=torch.bool)
    if alpha == 1.0:
        sigma_alpha = sigma.clone()
        transform = torch.eye(sigma.numel(), dtype=dtype, device=sigma.device)
    else:
        sigma_alpha, significant = _sigma_alpha_from_spectrum(
            sigma, alpha, rank_relative_tolerance)
        ratios = torch.zeros_like(sigma)
        ratios[significant] = sigma_alpha[significant] / sigma[significant]
        transform = (v * ratios.unsqueeze(0)) @ v.transpose(0, 1)
    ner = analytic_ner_from_spectrum(sigma_alpha, alpha=1.0, token_dim=token_dim,
                                     rank_relative_tolerance=rank_relative_tolerance,
                                     epsilon=epsilon)
    return {
        "alpha": float(alpha),
        "T": transform.cpu(),
        "sigma_alpha": sigma_alpha.cpu(),
        "significant": significant.cpu(),
        "analytic_ner": ner,
        "n_significant": int(significant.sum()),
        "nuclear_norm": float(sigma_alpha.sum()),
    }


def apply_spectral_transform(states: torch.Tensor, mask: Optional[torch.Tensor],
                             mu_ref: torch.Tensor, transform: torch.Tensor,
                             alpha: float = 1.0) -> torch.Tensor:
    """Apply mu_ref + (h - mu_ref) T_alpha to valid S-token positions.

    Supports [N, D] and [B, L, D]. Padding output is always zero. alpha=1 is an
    exact identity branch for valid tokens.
    """
    squeeze = False
    if states.dim() == 2:
        states_3d = states.unsqueeze(0)
        mask_3d = mask.unsqueeze(0) if mask is not None else None
        squeeze = True
    elif states.dim() == 3:
        states_3d = states
        mask_3d = mask
    else:
        raise ValueError("states must be [N, D] or [B, L, D]")

    if mask_3d is None:
        mask_3d = torch.ones(states_3d.shape[:2], dtype=states_3d.dtype,
                             device=states_3d.device)
    if alpha == 1.0:
        out = states_3d * mask_3d.unsqueeze(-1)
        return out.squeeze(0) if squeeze else out

    compute_dtype = torch.float64 if states_3d.dtype == torch.float64 else torch.float32
    x = states_3d.to(compute_dtype)
    mu = mu_ref.to(device=x.device, dtype=compute_dtype)
    t = transform.to(device=x.device, dtype=compute_dtype)
    transformed = mu + torch.matmul(x - mu, t)
    transformed = transformed * mask_3d.unsqueeze(-1).to(compute_dtype)
    out = transformed.to(states_3d.dtype)
    return out.squeeze(0) if squeeze else out
