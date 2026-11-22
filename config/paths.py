from typing import Any, Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG: Dict[str, Any] = {
    "project_root": PROJECT_ROOT,
    "data_root": PROJECT_ROOT / "data",
    "raw_root": PROJECT_ROOT / "data" / "raw",
    "interim_root": PROJECT_ROOT / "data" / "interim",
    "processed_root": PROJECT_ROOT / "data" / "processed",
    "metadata_root": PROJECT_ROOT / "data" / "metadata",
    "results_root": PROJECT_ROOT / "results",
    "checkpoints_root": PROJECT_ROOT / "results" / "checkpoints",
    "logs_root": PROJECT_ROOT / "results" / "logs",
    "metrics_root": PROJECT_ROOT / "results" / "metrics",
    "figures_root": PROJECT_ROOT / "results" / "figures",
    "raw_dataset_dirs": {
        "1k": Path("/Users/york/Documents/GitHub/hand-roll/data/raw_data/KuaiRand-1K"),
        "27k": Path("/Users/york/Documents/GitHub/hand-roll/data/raw_data/KuaiRand-27K"),
    },
    "processed_dataset_dirs": {
        "1k": PROJECT_ROOT / "data" / "processed" / "KuaiRand-1K",
        "27k": PROJECT_ROOT / "data" / "processed" / "KuaiRand-27K",
    },
    "unirank_reference": {
        "repo_url": "https://github.com/salmon1802/UniRank",
        "commit_hash": "90bb99a60724e975ec565836266fd86dcad0039d",
        "onetrans_file": "model_zoo/OneTrans.py",
        "preprocess_file": "data/KuaiRand/preprocess_Kuairand_seq_action.py",
        "tokenizer_file": "unirank/pytorch/layers/tokenization/unified_tokenizer.py",
        "local_impl_version": "1.0.0",
    },
}
