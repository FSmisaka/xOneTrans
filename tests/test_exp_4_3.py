"""Config, wrapper and continuous representation model checks for exp 4_3."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.exp_4_3_continuous_representation import CONFIG
from src.models.continuous_spectral_onetrans import ContinuousSpectralOneTrans
from src.models.spectral_transform import (apply_spectral_transform,
                                           build_spectral_transform,
                                           fit_spectral_basis)
from tests.test_exp_3 import _batch, _cfg, _vocab


def _basis():
    return fit_spectral_basis(
        torch.randn(240, 16, dtype=torch.float64), svd_dtype="float64",
        metadata={"calibration_split": "train", "rank_relative_tolerance": 1e-6})


def _transforms(basis):
    alphas = [float(a) for a in CONFIG["spectral"]["alpha_nodes"]]
    return [build_spectral_transform(basis["sigma"], basis["V"], a, token_dim=16)
            for a in alphas]


def _model():
    model = ContinuousSpectralOneTrans(
        _vocab(), _cfg(),
        {"kind": "continuous_representation_mix", "fusion": "contextual"},
        CONFIG["spectral"], gate_hidden_dim=8)
    basis = _basis()
    model.set_spectral_basis(basis, _transforms(basis),
                             CONFIG["spectral"]["alpha_nodes"],
                             CONFIG["spectral"]["quadrature_weights"])
    model.freeze_prefix()
    return model, basis


def test_continuous_config():
    cfg = CONFIG["spectral"]["continuous"]
    assert cfg["alpha_min"] == 0.125
    assert cfg["alpha_max"] == 1.0
    assert cfg["num_nodes"] >= 8
    assert cfg["quadrature"] == "gauss_legendre"
    assert "continuous_representation_mix" in CONFIG["conditions"]
    assert "continuous_representation_mix" in CONFIG["default_conditions"]
    assert all(float(a) <= 1.0 for a in CONFIG["spectral"]["alpha_nodes"])
    weights = [float(q) for q in CONFIG["spectral"]["quadrature_weights"]]
    assert abs(sum(weights) - 1.0) < 1e-6


def test_wrapper_applies_config_and_validates():
    import src.experiments.continuous_spectral as base
    import src.experiments.continuous_spectral_representation as wrapper

    old = (base.EXP_4, base.EXP_ID, base.EXP4_ALPHAS)
    try:
        wrapper._apply_config()
        assert base.EXP_4 is CONFIG
        assert base.EXP_ID == CONFIG["experiment"]
        assert base.EXP4_ALPHAS == [float(a) for a in CONFIG["spectral"]["alpha_nodes"]]
        base._validate_spectral_config()
    finally:
        base.EXP_4, base.EXP_ID, base.EXP4_ALPHAS = old


def test_continuous_model_matches_expected_transform_mixture():
    model, basis = _model()
    model.eval()
    batch = _batch(4, 7, 0)
    with torch.no_grad():
        out = model(batch, return_states=True)
    q = int(CONFIG["spectral"]["continuous"]["num_nodes"])
    assert out["weights"].shape == (4, q)
    assert torch.allclose(out["weights"].sum(-1), torch.ones(4), atol=1e-6)
    assert (out["weights"] >= 0).all()
    assert out["alpha_nodes"].numel() == q
    assert "logits_by_alpha" not in out
    assert "s_alpha" not in out
    assert out["s_mix"].shape == (4, 7, 16)
    assert out["s_final_mix"].shape == (4, 7, 16)

    alphas = out["alpha_nodes"].detach().cpu().tolist()
    transformed = []
    for alpha in alphas:
        transform = build_spectral_transform(basis["sigma"], basis["V"], alpha,
                                             token_dim=16)
        transformed.append(apply_spectral_transform(
            out["s_pre"], out["hist_mask"], basis["mu_ref"], transform["T"], alpha=alpha))
    manual_mix = torch.einsum("bq,bqld->bld", out["weights"],
                              torch.stack(transformed, dim=1))
    assert torch.allclose(out["s_mix"], manual_mix, atol=1e-5)


def test_continuous_model_gradients():
    model, _ = _model()
    model.train()
    batch = _batch(4, 7, 1)
    out = model(batch)
    loss = F.binary_cross_entropy(out["mix_prob"].clamp(1e-7, 1 - 1e-7), batch["label"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all()
                             for p in trainable)
    assert all(p.grad is None for p in frozen)
    assert any(p.grad.abs().sum() > 0 for p in model.context_gate.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.base.tower.parameters())


def test_aggregate_ner_handles_empty_node_lists():
    import src.experiments.continuous_spectral as base
    fake = {
        "ner": {
            "alpha_nodes": [0.125, 0.5],
            "pre_final": {"valid": [], "test": []},
            "post_final": {"valid": [], "test": []},
            "s_pre": {"valid": 0.30, "test": 0.31},
            "representation_mix": {"valid_pre": 0.40, "valid_post": 0.41,
                                   "test_pre": 0.42, "test_post": 0.43},
        },
        "reference_ner": [0.8, 0.6],
        "calibration_ner": [0.8, 0.6],
        "spectral_alpha_nodes": [0.125, 0.5],
    }
    agg = base._aggregate_ner([fake])
    assert agg["pre_final"]["valid"] == []
    assert agg["post_final"]["test"] == []
    assert agg["representation_mix"]["valid_pre"]["mean"] == 0.40


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"[done] {len(tests)} checks passed")


if __name__ == "__main__":
    main()
