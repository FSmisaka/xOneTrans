"""Numerical and structural checks for SPEC4 (continuous spectral OneTrans)."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.exp_4_continuous_spectral import CONFIG as EXP_4
from src.experiments.diagnostics import normalized_effective_rank
from src.models.continuous_spectral_onetrans import (ContinuousSpectralOneTrans,
                                                     probability_mix, simpson_weights)
from src.models.onetrans_diagnostic import OneTransDiagnostic
from src.models.spectral_transform import (analytic_ner_from_spectrum,
                                           apply_spectral_transform,
                                           build_spectral_transform,
                                           fit_spectral_basis)
from tests.test_exp_3 import _batch, _cfg, _vocab


def _basis(seed=0, n=300, d=16):
    torch.manual_seed(seed)
    x = torch.randn(n, d, dtype=torch.float64) * torch.randn(d, dtype=torch.float64)
    return fit_spectral_basis(x, svd_dtype="float64", metadata={"calibration_split": "train"})


def _transforms(basis, alphas=None):
    alphas = alphas or EXP_4["spectral"]["alpha_nodes"]
    return [build_spectral_transform(basis["sigma"], basis["V"], float(a),
                                     token_dim=basis["sigma"].numel())
            for a in alphas]


def test_identity_transform_and_padding():
    basis = _basis()
    alphas = EXP_4["spectral"]["alpha_nodes"]
    tr = _transforms(basis, alphas)[2]
    states = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7)
    mask[0, 4:] = 0
    mask[1, :] = 0
    states[0, 4:] = 9.0
    states[1, :] = 7.0
    out = apply_spectral_transform(states, mask, basis["mu_ref"], tr["T"], alpha=1.0)
    assert torch.equal(out[0, :4], states[0, :4])
    assert torch.equal(out[2], states[2])
    assert out[~mask.bool()].abs().sum() == 0
    x2 = states[0, :4]
    out2 = apply_spectral_transform(x2, None, basis["mu_ref"], tr["T"], alpha=1.0)
    assert torch.equal(out2, x2)


def test_reference_spectrum_matches_analytic_ner():
    torch.manual_seed(11)
    x_ref = torch.randn(500, 16, dtype=torch.float64) * torch.randn(16, dtype=torch.float64)
    basis = fit_spectral_basis(x_ref, svd_dtype="float64",
                               metadata={"calibration_split": "train"})
    trs = _transforms(basis)
    for tr in trs:
        transformed = apply_spectral_transform(x_ref, None, basis["mu_ref"], tr["T"],
                                               alpha=tr["alpha"])
        empirical = normalized_effective_rank(transformed)
        assert abs(empirical - tr["analytic_ner"]) < 1e-4


def test_alpha_monotonicity():
    sigma = torch.linspace(5.0, 0.05, 32, dtype=torch.float64)
    alphas = [0.25, 0.5, 0.70710678, 1.0, 1.41421356, 2.0, 4.0]
    ners = [analytic_ner_from_spectrum(sigma, a, token_dim=32) for a in alphas]
    assert all(ners[i] >= ners[i + 1] - 1e-12 for i in range(len(ners) - 1))
    assert ners[2] >= ners[3]
    assert ners[4] <= ners[3]


def test_nuclear_norm_is_preserved():
    basis = _basis(seed=2)
    total = float(basis["sigma"].sum())
    for tr in _transforms(basis):
        assert abs(float(tr["sigma_alpha"].sum()) - total) / total < 1e-4


def test_batch_independence():
    basis = _basis(seed=3)
    tr = _transforms(basis)[0]
    states = torch.randn(4, 7, 16)
    mask = torch.ones(4, 7)
    mask[1, 5:] = 0
    mask[2, :] = 0
    batch = apply_spectral_transform(states, mask, basis["mu_ref"], tr["T"], alpha=tr["alpha"])
    singles = torch.cat([
        apply_spectral_transform(states[i:i + 1], mask[i:i + 1], basis["mu_ref"],
                                 tr["T"], alpha=tr["alpha"])
        for i in range(4)], dim=0)
    assert torch.allclose(batch, singles, atol=1e-6)


def test_padding_invariance():
    basis = _basis(seed=4)
    tr = _transforms(basis)[0]
    states = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7)
    mask[0, 4:] = 0
    mask[1, 6:] = 0
    changed = states.clone()
    changed[~mask.bool()] = 123.0
    a = apply_spectral_transform(states, mask, basis["mu_ref"], tr["T"], alpha=tr["alpha"])
    b = apply_spectral_transform(changed, mask, basis["mu_ref"], tr["T"], alpha=tr["alpha"])
    assert torch.allclose(a, b, atol=1e-6)
    assert b[~mask.bool()].abs().sum() == 0


def test_backbone_prefix_final_equivalence():
    model = OneTransDiagnostic(_vocab(), _cfg()).eval()
    batch = _batch(4, 7, 0)
    s_tokens, ns_tokens, mask = model.encode_inputs(batch)
    with torch.no_grad():
        ns_full, s_full = model.backbone(s_tokens, ns_tokens, mask)
        s_pre, ns_pre, mask_pre = model.backbone.forward_prefix(
            s_tokens, ns_tokens, mask, end_layer=model.backbone.num_layers - 1)
        s_final, ns_final, mask_final = model.backbone.forward_one_layer(
            s_pre, ns_pre, mask_pre, layer_idx=model.backbone.num_layers - 1)
        logit_full = model(batch)["logit"]
        logit_parts = model.final_layer(s_pre, ns_pre, mask_pre)["logit"]
    assert torch.allclose(ns_full, ns_final, atol=1e-6)
    assert torch.allclose(s_full, s_final, atol=1e-6)
    assert torch.allclose(logit_full, logit_parts, atol=1e-6)
    assert torch.equal(mask_pre, mask)
    assert torch.equal(mask_final, mask)


def _continuous_model(kind, fusion, seed=0):
    basis = _basis(seed=seed)
    trs = _transforms(basis)
    cond = {"kind": kind}
    if fusion is not None:
        cond["fusion"] = fusion
    if kind == "single":
        cond["alpha"] = 1.0
    model = ContinuousSpectralOneTrans(_vocab(), _cfg(), cond, EXP_4["spectral"],
                                       gate_hidden_dim=8)
    model.set_spectral_basis(basis, trs, EXP_4["spectral"]["alpha_nodes"],
                             EXP_4["spectral"]["quadrature_weights"])
    model.freeze_prefix()
    return model, basis


def test_shared_parameters_and_gate_weights():
    model, _ = _continuous_model("expert_mix", "uniform")
    base_params = sum(p.numel() for p in model.base.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert total < base_params * 1.2, "K independent OneTrans copies detected"
    assert len(model.base.backbone.mha_layers) == _cfg()["num_layers"]
    model.eval()
    out = model(_batch(4, 7, 1))
    assert out["logits_by_alpha"].shape[1] == len(EXP_4["spectral"]["alpha_nodes"])
    assert out["weights"].shape == out["logits_by_alpha"].shape
    assert torch.allclose(out["weights"].sum(-1), torch.ones(4), atol=1e-6)
    assert (out["weights"] >= 0).all()
    q = torch.tensor(EXP_4["spectral"]["quadrature_weights"])
    assert torch.allclose(out["weights"][0], q, atol=1e-6)

    scores = torch.zeros(3, 7)
    w = simpson_weights(scores, q)
    assert torch.allclose(w, q.expand(3, -1), atol=1e-6)
    probs = torch.rand(3, 7)
    one_hot = torch.zeros(3, 7)
    one_hot[:, 3] = 1.0
    assert torch.allclose(probability_mix(probs, one_hot), probs[:, 3])

    for fusion in ("global", "contextual"):
        model, _ = _continuous_model("expert_mix", fusion)
        model.eval()
        out = model(_batch(4, 7, 2))
        assert torch.allclose(out["weights"].sum(-1), torch.ones(4), atol=1e-6)
        assert (out["weights"] >= 0).all()


def test_gradients_and_frozen_prefix():
    batch = _batch(4, 7, 3)
    for kind, fusion in (("expert_mix", "uniform"),
                         ("expert_mix", "contextual"),
                         ("representation_mix", "contextual")):
        model, _ = _continuous_model(kind, fusion, seed=5)
        model.train()
        out = model(batch)
        if kind == "expert_mix":
            logits = out["logits_by_alpha"]
            q = out["quad_weights"]
            loss = F.binary_cross_entropy(out["mix_prob"].clamp(1e-7, 1 - 1e-7),
                                          batch["label"])
            loss = loss + 0.1 * sum(
                q[k] * F.binary_cross_entropy_with_logits(logits[:, k], batch["label"])
                for k in range(logits.shape[1]))
        else:
            loss = F.binary_cross_entropy_with_logits(out["logit"], batch["label"])
        model.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all()
                                 for p in trainable)
        assert all(p.grad is None for p in frozen)
        assert any(p.grad.abs().sum() > 0 for p in model.base.tower.parameters())
        assert any(p.grad.abs().sum() > 0 for p in model.base.backbone.ffn_layers[-1].parameters())
        if fusion == "contextual":
            gate = model.context_gate
            assert any(p.grad.abs().sum() > 0 for p in gate.parameters())


def test_no_calibration_leakage_metadata():
    assert EXP_4["calibration"]["split"] == "train"
    tokens = torch.randn(200, 16)
    basis = fit_spectral_basis(tokens, svd_dtype="float64",
                               metadata={"calibration_split": "train"})
    assert basis["metadata"]["calibration_split"] == "train"
    from src.experiments.continuous_spectral import build_basis_payload
    calibration = {"tokens": tokens, "sample_indices": [1, 2, 3],
                   "n_tokens": int(tokens.shape[0]), "n_samples": 3}
    payload = build_basis_payload(calibration, "1k", 2026, Path("dummy.pt"), "sig", True)
    assert payload["metadata"]["calibration_split"] == "train"
    assert payload["metadata"]["sample_indices"] == [1, 2, 3]


def test_zero_and_invalid_calibration_raise():
    for bad in (torch.empty(0, 16), torch.zeros(20, 16),
                torch.full((20, 16), float("nan"))):
        try:
            fit_spectral_basis(bad)
        except ValueError:
            continue
        raise AssertionError("invalid calibration matrix did not raise")


def test_low_rank_spectrum_is_finite():
    torch.manual_seed(6)
    for x in (torch.randn(200, 16, dtype=torch.float64),
              torch.randn(8, 16, dtype=torch.float64)):
        if x.shape[0] > 100:
            x = x.clone()
            x[:, 10:] = 0.0
        basis = fit_spectral_basis(x, svd_dtype="float64")
        for tr in _transforms(basis):
            transformed = apply_spectral_transform(x, None, basis["mu_ref"], tr["T"],
                                                   alpha=tr["alpha"])
            assert abs(normalized_effective_rank(transformed) - tr["analytic_ner"]) < 1e-4
            assert np.isfinite(tr["analytic_ner"])
            assert abs(float(tr["sigma_alpha"].sum()) - float(basis["sigma"].sum())) < 1e-6


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"[done] {len(tests)} checks passed")


if __name__ == "__main__":
    main()
