"""Structural and numerical checks for the SPEC3 dual-branch experiment."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import VARIABLE_FIELDS
from src.experiments.diagnostics import normalized_effective_rank
from src.models.dual_branch_onetrans import DualBranchOneTrans
from src.models.ner_regularization import differentiable_ner_stats
from src.models.onetrans_diagnostic import OneTransDiagnostic


def _vocab():
    return {"item_id": 51, "action": 64,
            "variable_fields": {f: 7 for f in VARIABLE_FIELDS}}


def _cfg():
    return {
        "embedding_dim": 8, "token_dim": 16, "num_layers": 2, "num_heads": 2,
        "num_ns_token": 2, "tokenizer_type": "Auto", "tokenizer_activation": "SiLU",
        "reduction_ratio": 0.0, "expansion_factor": 2, "dnn_activations": "ReLU",
        "tower_hidden_units": [16], "net_dropout": 0.0, "max_history_len": 7,
        "use_residual_gate": False,
        "mix_head": {"hidden_dim": None, "dropout": 0.0, "layer_norm": False},
    }


def _batch(batch_size=6, length=7, seed=0):
    torch.manual_seed(seed)
    batch = {
        "hist_item": torch.randint(1, 51, (batch_size, length)),
        "hist_action": torch.randint(0, 64, (batch_size, length)),
        "hist_mask": torch.ones(batch_size, length),
        "item_id": torch.randint(1, 51, (batch_size,)),
        "label": torch.randint(0, 2, (batch_size,)).float(),
    }
    for field in VARIABLE_FIELDS:
        batch[field] = torch.randint(0, 7, (batch_size,))
    batch["hist_mask"][0, 3:] = 0.0
    batch["hist_mask"][1, :] = 0.0
    return batch


def test_shapes_and_independent_parameters():
    model = DualBranchOneTrans(_vocab(), _cfg()).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch, return_sequence_states=True)
    b = batch["label"].shape[0]
    for key in ("logit_high", "logit_low", "logit_mix", "logit_avg"):
        assert out[key].shape == (b,), f"{key}: {out[key].shape}"
    assert out["readout_high"].shape[-1] == model.readout_dim
    assert out["readout_low"].shape[-1] == model.readout_dim
    assert len(out["states_high"]) == _cfg()["num_layers"] + 1
    assert len(out["states_low"]) == _cfg()["num_layers"] + 1

    ids_high = {id(p) for p in model.high_branch.parameters()}
    ids_low = {id(p) for p in model.low_branch.parameters()}
    assert ids_high.isdisjoint(ids_low), "branches share parameter objects"
    state_high = model.high_branch.state_dict()
    state_low = model.low_branch.state_dict()
    assert state_high.keys() == state_low.keys()
    assert all(torch.equal(state_high[k], state_low[k]) for k in state_high), \
        "branches do not start from the same initialization"

    mix_first, mix_last = model.mix_head.net[0], model.mix_head.net[-1]
    assert mix_first.in_features == 2 * model.readout_dim
    assert mix_first.out_features == model.readout_dim
    assert mix_last.out_features == 1


def test_single_branch_forward_still_works():
    model = OneTransDiagnostic(_vocab(), _cfg()).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch, return_readout=True, return_last_state=True)
    assert out["logit"].shape == (batch["label"].shape[0],)
    assert out["readout"].shape[-1] == _cfg()["token_dim"] * _cfg()["num_ns_token"]
    assert out["last_state"].shape[:2] == batch["hist_item"].shape


def test_lambda_zero_ner_does_not_change_gradients():
    model = DualBranchOneTrans(_vocab(), _cfg()).eval()
    batch = _batch()
    label = batch["label"]

    def ctr_loss(out):
        return (F.binary_cross_entropy_with_logits(out["logit_high"], label)
                + F.binary_cross_entropy_with_logits(out["logit_low"], label)
                + F.binary_cross_entropy_with_logits(out["logit_mix"], label)) / 3.0

    model.zero_grad(set_to_none=True)
    ctr_loss(model(batch)).backward()
    grads_a = {n: p.grad.detach().clone() for n, p in model.named_parameters()
               if p.grad is not None}

    model.zero_grad(set_to_none=True)
    out = model(batch)
    stats_h = differentiable_ner_stats(out["last_state_high"], out["hist_mask"])
    stats_l = differentiable_ner_stats(out["last_state_low"], out["hist_mask"])
    (ctr_loss(out) + 0.0 * (-stats_h["h_norm"]) + 0.0 * stats_l["h_norm"]).backward()
    grads_b = {n: p.grad.detach().clone() for n, p in model.named_parameters()
               if p.grad is not None}
    assert grads_a.keys() == grads_b.keys()
    assert all(torch.allclose(grads_a[n], grads_b[n], atol=1e-7) for n in grads_a)


def test_padding_does_not_change_predictions():
    model = DualBranchOneTrans(_vocab(), _cfg()).eval()
    batch = _batch()
    with torch.no_grad():
        out_a = model(batch)
        padded = batch.copy()
        padded["hist_item"] = batch["hist_item"].clone()
        padded["hist_action"] = batch["hist_action"].clone()
        pad = batch["hist_mask"] == 0
        padded["hist_item"][pad] = 3
        padded["hist_action"][pad] = 1
        out_b = model(padded)
    for key in ("logit_high", "logit_low", "logit_mix", "logit_avg"):
        assert torch.allclose(out_a[key], out_b[key], atol=1e-6), key
    assert torch.isfinite(out_a["logit_mix"]).all()


def test_differentiable_ner_matches_evaluation_ner():
    torch.manual_seed(0)
    for shape in [(5000, 16), (16, 5000), (100, 16), (200, 32)]:
        x = torch.randn(*shape) * torch.randn(shape[1])
        ours = differentiable_ner_stats(x)["ner"].item()
        ref = normalized_effective_rank(x)
        assert abs(ours - ref) < 1e-4, f"{shape}: ours={ours}, ref={ref}"

        low_rank = x.clone()
        if shape[1] > 8:
            low_rank[:, 8:] = 0.0
        ours = differentiable_ner_stats(low_rank)["ner"].item()
        ref = normalized_effective_rank(low_rank)
        assert abs(ours - ref) < 1e-4, f"rank-deficient {shape}: {ours} vs {ref}"


def test_padding_and_empty_history_are_excluded():
    torch.manual_seed(1)
    states = torch.randn(5, 7, 16)
    mask = torch.ones(5, 7)
    mask[0, 3:] = 0.0
    mask[1, :] = 0.0
    masked = differentiable_ner_stats(states, mask)
    flat = differentiable_ner_stats(states[mask.bool()])
    assert abs(masked["ner"].item() - flat["ner"].item()) < 1e-6
    assert masked["n_tokens"] == int(mask.sum())

    garbage = states.clone()
    garbage[~mask.bool()] = 1e3
    assert abs(differentiable_ner_stats(garbage, mask)["ner"].item()
               - masked["ner"].item()) < 1e-6

    empty = differentiable_ner_stats(states, torch.zeros_like(mask))
    assert empty["ner"].item() == 0.0
    assert torch.isfinite(empty["h_norm"])


def test_ner_regularizer_gradient_direction():
    torch.manual_seed(2)
    base = (torch.randn(300, 1) @ torch.randn(1, 16)
            + 0.05 * torch.randn(300, 16))

    def run(sign):
        x = base.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=5e-3)
        before = differentiable_ner_stats(x)["ner"].item()
        for _ in range(250):
            stats = differentiable_ner_stats(x)
            loss = sign * stats["h_norm"]
            opt.zero_grad()
            loss.backward()
            opt.step()
        after = differentiable_ner_stats(x)["ner"].item()
        return before, after

    before, after = run(-1.0)
    assert after > before + 0.05, f"maximizing NER failed: {before} -> {after}"
    before, after = run(1.0)
    assert after < before - 0.05, f"minimizing NER failed: {before} -> {after}"


def test_model_ner_optimization_direction():
    batch = _batch(32, 7, seed=3)

    def run(sign):
        torch.manual_seed(0)
        model = DualBranchOneTrans(_vocab(), _cfg()).eval()

        def current_ner():
            with torch.no_grad():
                out = model(batch)
                return differentiable_ner_stats(
                    out["last_state_high"], out["hist_mask"])["ner"].item()

        before = current_ner()
        opt = torch.optim.Adam(model.high_branch.parameters(), lr=1e-3)
        for _ in range(30):
            out = model(batch)
            stats = differentiable_ner_stats(out["last_state_high"], out["hist_mask"])
            opt.zero_grad()
            (sign * stats["h_norm"]).backward()
            opt.step()
        return before, current_ner()

    before, after = run(-1.0)
    assert after > before + 0.02, f"model high-NER direction: {before} -> {after}"
    before, after = run(1.0)
    assert after < before - 0.02, f"model low-NER direction: {before} -> {after}"


def test_mixed_loss_reaches_both_branches():
    model = DualBranchOneTrans(_vocab(), _cfg()).eval()
    batch = _batch()
    model.zero_grad(set_to_none=True)
    out = model(batch)
    F.binary_cross_entropy_with_logits(out["logit_mix"], batch["label"]).backward()
    for branch in (model.high_branch, model.low_branch):
        grad = branch.item_embedding.weight.grad
        assert grad is not None and grad.abs().sum() > 0
    assert model.mix_head.net[0].weight.grad is not None


def test_branch_ner_loss_is_isolated():
    model = DualBranchOneTrans(_vocab(), _cfg()).eval()
    batch = _batch()
    model.zero_grad(set_to_none=True)
    out = model(batch)
    stats_h = differentiable_ner_stats(out["last_state_high"], out["hist_mask"])
    (-stats_h["h_norm"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.high_branch.parameters())
    assert all(p.grad is None for p in model.low_branch.parameters())
    assert all(p.grad is None for p in model.mix_head.parameters())


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"[done] {len(tests)} checks passed")


if __name__ == "__main__":
    main()
