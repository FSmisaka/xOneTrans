from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_2_1_task_aware_gate",
    "seeds": [2026, 2027, 2028],
    "residual_gate": {
        "residual_hidden_dim": 128,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.0,
        "gate_init_bias": 0.0,
    },
    "evaluation": {"history_shuffle": True},
}
