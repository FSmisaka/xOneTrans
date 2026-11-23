from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_2_2_gate_ablation",
    "seeds": [2026, 2027, 2028],
    "residual_gate": {
        "residual_hidden_dim": 128,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.0,
        "gate_init_bias": 0.0,
    },
    "fixed_gate_alphas": [0.25, 0.5, 0.75],
    "evaluation": {"history_shuffle": True},
}
