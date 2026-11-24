"""Config, wrapper and 7-node model checks for exp 4_2 (low-alpha)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.exp_4_2_low_alpha import CONFIG
from src.models.continuous_spectral_onetrans import ContinuousSpectralOneTrans
from src.models.spectral_transform import build_spectral_transform, fit_spectral_basis
from tests.test_exp_3 import _batch, _cfg, _vocab


def test_low_alpha_grid_and_log_spacing():
    alphas = [float(a) for a in CONFIG["spectral"]["alpha_nodes"]]
    assert len(alphas) == 7
    assert all(0.0 < a <= 1.0 for a in alphas)
    assert alphas[-1] == 1.0
    assert all(alphas[i] < alphas[i + 1] for i in range(len(alphas) - 1))
    log_diffs = np.diff(np.log(alphas))
    assert np.max(log_diffs) - np.min(log_diffs) < 1e-6, "nodes should be log-alpha spaced"


def test_quadrature_weights():
    weights = [float(q) for q in CONFIG["spectral"]["quadrature_weights"]]
    assert len(weights) == len(CONFIG["spectral"]["alpha_nodes"])
    assert all(q >= 0 for q in weights)
    assert abs(sum(weights) - 1.0) < 1e-6


def test_default_conditions_cover_single_and_mixture():
    default = CONFIG["default_conditions"]
    assert "baseline_frozen_prefix" in default
    for alpha in CONFIG["spectral"]["alpha_nodes"]:
        if float(alpha) < 1.0:
            assert f"fixed_alpha_{float(alpha):g}" in default or any(
                name.startswith("fixed_alpha_") and abs(float(name.split("_")[-1]) - float(alpha)) < 1e-8
                for name in default)
    for mixture in ("uniform_expert_mix", "global_learned_expert_mix",
                    "contextual_learned_expert_mix", "contextual_representation_mix"):
        assert mixture in default


def test_wrapper_applies_config_and_validates():
    import src.experiments.continuous_spectral as base
    import src.experiments.continuous_spectral_low_alpha as wrapper

    old = (base.EXP_4, base.EXP_ID, base.EXP4_ALPHAS)
    try:
        wrapper._apply_config()
        assert base.EXP_4 is CONFIG
        assert base.EXP_ID == CONFIG["experiment"]
        assert base.EXP4_ALPHAS == [float(a) for a in CONFIG["spectral"]["alpha_nodes"]]
        base._validate_spectral_config()
    finally:
        base.EXP_4, base.EXP_ID, base.EXP4_ALPHAS = old


def test_low_alpha_model_forward():
    alphas = [float(a) for a in CONFIG["spectral"]["alpha_nodes"]]
    basis = fit_spectral_basis(
        torch.randn(200, 16, dtype=torch.float64), svd_dtype="float64",
        metadata={"calibration_split": "train"})
    transforms = [build_spectral_transform(basis["sigma"], basis["V"], alpha,
                                           token_dim=16) for alpha in alphas]
    model = ContinuousSpectralOneTrans(
        _vocab(), _cfg(), {"kind": "expert_mix", "fusion": "uniform"},
        CONFIG["spectral"], gate_hidden_dim=8)
    model.set_spectral_basis(basis, transforms, alphas,
                             CONFIG["spectral"]["quadrature_weights"])
    model.freeze_prefix()
    model.eval()
    out = model(_batch(4, 7, 0))
    assert out["logits_by_alpha"].shape == (4, 7)
    assert out["weights"].shape == (4, 7)
    assert torch.allclose(out["weights"].sum(-1), torch.ones(4), atol=1e-6)
    q = torch.tensor(CONFIG["spectral"]["quadrature_weights"], dtype=torch.float32)
    assert torch.allclose(out["weights"][0], q, atol=1e-6)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"[done] {len(tests)} checks passed")


if __name__ == "__main__":
    main()
