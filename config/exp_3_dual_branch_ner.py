from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_3_dual_branch_ner",
    "seeds": [2026, 2027, 2028],
    "ner_lambda": 1e-2,
    "conditions": {
        "single_baseline": {"mode": "single"},
        "dual_control": {"mode": "dual", "lambda_high": 0.0, "lambda_low": 0.0},
        "high_only": {"mode": "dual", "lambda_high": "ner_lambda", "lambda_low": 0.0},
        "low_only": {"mode": "dual", "lambda_high": 0.0, "lambda_low": "ner_lambda"},
        "dual_split_ner": {"mode": "dual", "lambda_high": "ner_lambda",
                           "lambda_low": "ner_lambda"},
    },
    "default_conditions": ["single_baseline", "dual_control", "high_only",
                           "low_only", "dual_split_ner"],
    "lambda_sweep": [0.0, 1e-3, 1e-2, 1e-1],
    "ner_target_layers": ["last"],
    "ner": {
        "warmup_ratio": 0.05,
        "epsilon": 1e-8,
        "spectral_backend": "auto",
    },
    "mix_head": {"hidden_dim": None, "dropout": None, "layer_norm": False},
    "training": {"accum_steps_multiplier": 2},
    "evaluation": {"report_logit_avg": True, "compute_test_ner": True},
    "analysis": {"compute_paired_deltas": True},
    "debug": {
        "epochs": 1,
        "train_samples": 4096,
        "valid_samples": 2048,
        "test_samples": 2048,
        "diagnostic_samples": 512,
        "ner_max_tokens": 1024,
    },
}
