from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_2_3_gate_analysis",
    "seeds": [2026, 2027, 2028],
    "residual_gate": {
        "residual_hidden_dim": 128,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.0,
        "gate_init_bias": 0.0,
    },
    "shuffle_seed": 2026,
    "history_length_buckets": [1, 5, 20, 50, 101],
    "group_field": "duration_bucket",
    "evaluation": {
        "compute_effective_rank": True,
        "compute_history_shuffle": True,
        "compute_gate_statistics": True,
        "compute_gate_intervention": True,
    },
}
