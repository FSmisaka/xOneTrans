from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_1_3_type",
    "conditions": {
        "User": {"user": 4, "context": 0, "item": 0},
        "Context": {"user": 0, "context": 4, "item": 0},
        "TargetItem": {"user": 0, "context": 0, "item": 4},
        "All": {"user": 4, "context": 4, "item": 4},
        "None": {"user": 0, "context": 0, "item": 0},
    },
}
