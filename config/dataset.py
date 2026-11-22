from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "dataset_names": {"1k": "KuaiRand-1K", "27k": "KuaiRand-27K"},
    "log_files": {
        "1k": [
            "log_random_4_22_to_5_08_1k.csv",
            "log_standard_4_08_to_4_21_1k.csv",
            "log_standard_4_22_to_5_08_1k.csv",
        ],
        "27k": [
            "log_random_4_22_to_5_08_27k.csv",
            "log_standard_4_08_to_4_21_27k_part1.csv",
            "log_standard_4_08_to_4_21_27k_part2.csv",
            "log_standard_4_22_to_5_08_27k_part1.csv",
            "log_standard_4_22_to_5_08_27k_part2.csv",
        ],
    },
    "user_features_file": {"1k": "user_features_1k.csv", "27k": "user_features_27k.csv"},
    "video_features_file": {"1k": "video_features_basic_1k.csv", "27k": "video_features_basic_27k.csv"},
    "log_columns": [
        "user_id", "video_id", "time_ms",
        "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
        "tab",
    ],
    "label_columns": ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view"],
    "user_profile_fields": [
        "user_active_degree", "follow_user_num_range", "fans_user_num_range", "register_days_range",
    ],
    "context_fields": ["tab", "day_of_week", "is_weekend", "hour"],
    "target_item_fields": ["video_type", "primary_tag", "music_type", "duration_bucket"],
    "variable_fields": {
        "user": ["user_active_degree", "follow_user_num_range", "fans_user_num_range", "register_days_range"],
        "context": ["tab", "day_of_week", "is_weekend", "hour"],
        "item": ["video_type", "primary_tag", "music_type", "duration_bucket"],
    },
    "timezone": "Asia/Shanghai",
    "min_user_interactions": 10,
    "min_feat_count": 2,
    "split_ratios": {"train": 0.8, "valid": 0.1, "test": 0.1},
    "sample_caps": {"train": 1_000_000, "valid": 100_000, "test": 100_000},
    "debug_sample_caps": {"train": 200_000, "valid": 20_000, "test": 20_000},
    "sampling_seed": 2026,
    "csv_chunk_size": 2_000_000,
    "n_user_partitions": {"1k": 8, "27k": 64},
    "partition_flush_rows": 500_000,
    "duration_bins": [-1, 0, 3000, 7000, 15000, 30000, 60000, float("inf")],
    "duration_labels": ["0", "0-3s", "3-7s", "7-15s", "15-30s", "30-60s", "60s+"],
}
