from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "probe_model": "LogisticRegression",
    "probe_C": 1.0,
    "probe_max_iter": 1000,
    "probe_solver": "lbfgs",
    "probe_train_max": 100_000,
    "probe_eval_max": 50_000,
    "diagnostic_samples": 20_000,
    "diagnostic_seed": 2026,
    "ner_max_tokens": 50_000,
    "ner_seed": 2026,
    "ner_dtype": "float32",
}
