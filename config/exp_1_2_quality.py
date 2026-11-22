from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_1_2_quality",
    "quality_levels": [1.0, 0.75, 0.5, 0.25, 0.0],
    "permutation_seed": 2026,
    "n_importance_permutations": 3,
    "importance_seed": 2027,
    "static_probe": {
        "hidden_units": [256, 128],
        "embedding_dim": 16,
        "batch_size": 4096,
        "epochs": 3,
        "learning_rate": 1e-3,
        "train_fit_ratio": 0.8,
        "seed": 2026,
    },
}
