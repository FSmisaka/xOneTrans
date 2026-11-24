from typing import Any, Dict

from config.paths import CONFIG as PATHS

BASELINE_CKPT = str(
    PATHS["results_root"] / "runs" / "exp_3_dual_branch_ner" / "single_baseline"
    / "seed_{seed}" / "checkpoint_best_mix.pt"
)

ALPHA_NODES = [0.5, 0.70710678, 1.0, 1.41421356, 2.0, 2.82842712, 4.0]
QUAD_WEIGHTS = [1 / 18, 4 / 18, 2 / 18, 4 / 18, 2 / 18, 4 / 18, 1 / 18]

CONFIG: Dict[str, Any] = {
    "experiment": "exp_4_continuous_spectral",
    "seeds": [2026, 2027, 2028],
    "baseline_checkpoint": BASELINE_CKPT,
    "dataset": "1k",
    "calibration": {
        "split": "train",
        "max_tokens": 50000,
        "seed": 2026,
        "rank_relative_tolerance": 1e-6,
        "svd_dtype": "float64",
        "batch_size": 1024,
    },
    "spectral": {
        "alpha_nodes": ALPHA_NODES,
        "quadrature_weights": QUAD_WEIGHTS,
        "integration_variable": "log_alpha",
    },
    "model": {
        "trainable_last_n_layers": 1,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.0,
        "fusion_modes": [
            "uniform_expert_mix",
            "global_learned_expert_mix",
            "contextual_learned_expert_mix",
            "contextual_representation_mix",
        ],
    },
    "training": {
        "node_aux_loss_weight": 0.1,
        "node_aux_loss_ablation": [0.0, 0.1],
        "freeze_prefix": True,
        "accum_steps_multiplier": 2,
    },
    "evaluation": {
        "compute_global_ner": True,
        "compute_node_metrics": True,
        "compute_weight_stats": True,
        "compute_cost": True,
        "diagnostic_samples": 20000,
        "ner_max_tokens": 50000,
        "ner_batch_size": 512,
        "history_length_buckets": [1, 5, 20, 50, 101],
        "group_field": "duration_bucket",
    },
    "conditions": {
        "baseline_frozen_prefix": {"kind": "single", "alpha": 1.0},
        "fixed_alpha_0.5": {"kind": "single", "alpha": 0.5},
        "fixed_alpha_0.70710678": {"kind": "single", "alpha": 0.70710678},
        "fixed_alpha_1.41421356": {"kind": "single", "alpha": 1.41421356},
        "fixed_alpha_2.0": {"kind": "single", "alpha": 2.0},
        "fixed_alpha_2.82842712": {"kind": "single", "alpha": 2.82842712},
        "fixed_alpha_4.0": {"kind": "single", "alpha": 4.0},
        "uniform_expert_mix": {
            "kind": "expert_mix", "fusion": "uniform", "node_aux_loss_weight": 0.1,
        },
        "global_learned_expert_mix": {
            "kind": "expert_mix", "fusion": "global", "node_aux_loss_weight": 0.1,
        },
        "contextual_learned_expert_mix": {
            "kind": "expert_mix", "fusion": "contextual", "node_aux_loss_weight": 0.1,
        },
        "contextual_representation_mix": {
            "kind": "representation_mix", "fusion": "contextual",
            "node_aux_loss_weight": 0.0,
        },
        "uniform_expert_mix_noaux": {
            "kind": "expert_mix", "fusion": "uniform", "node_aux_loss_weight": 0.0,
        },
        "global_learned_expert_mix_noaux": {
            "kind": "expert_mix", "fusion": "global", "node_aux_loss_weight": 0.0,
        },
        "contextual_learned_expert_mix_noaux": {
            "kind": "expert_mix", "fusion": "contextual", "node_aux_loss_weight": 0.0,
        },
    },
    "default_conditions": [
        "baseline_frozen_prefix",
        "fixed_alpha_0.5",
        "fixed_alpha_0.70710678",
        "fixed_alpha_1.41421356",
        "fixed_alpha_2.0",
        "fixed_alpha_2.82842712",
        "fixed_alpha_4.0",
        "uniform_expert_mix",
        "global_learned_expert_mix",
        "contextual_learned_expert_mix",
        "contextual_representation_mix",
    ],
    "ablation_conditions": [
        "uniform_expert_mix_noaux",
        "global_learned_expert_mix_noaux",
        "contextual_learned_expert_mix_noaux",
    ],
    "debug": {
        "epochs": 1,
        "train_samples": 4096,
        "valid_samples": 2048,
        "test_samples": 2048,
        "calibration_max_tokens": 2048,
        "diagnostic_samples": 512,
        "ner_max_tokens": 1024,
        "ner_batch_size": 256,
    },
}
