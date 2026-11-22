from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "experiment": "exp_1_1_count",
    "feature_counts": [0, 3, 6, 9, 12],
    "field_order": [
        ("user", "user_active_degree"),
        ("context", "tab"),
        ("item", "video_type"),
        ("user", "follow_user_num_range"),
        ("context", "day_of_week"),
        ("item", "primary_tag"),
        ("user", "fans_user_num_range"),
        ("context", "is_weekend"),
        ("item", "music_type"),
        ("user", "register_days_range"),
        ("context", "hour"),
        ("item", "duration_bucket"),
    ],
}
