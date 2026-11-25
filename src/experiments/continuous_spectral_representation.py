"""Exp4.3: continuous representation-level spectral fusion.

The gate defines a continuous density over log-alpha; a Gauss-Legendre rule
integrates the expected spectral transform T_bar, and the shared final block is
executed once. The upstream OneTrans prefix and spectral basis stay fixed.
"""

from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.exp_4_3_continuous_representation import CONFIG as EXP_4_3
import src.experiments.continuous_spectral as _base


def _apply_config() -> None:
    _base.EXP_4 = EXP_4_3
    _base.EXP_ID = EXP_4_3["experiment"]
    _base.EXP4_ALPHAS = [float(a) for a in EXP_4_3["spectral"]["alpha_nodes"]]


def run_experiment_4_3(dataset_key: str, seeds: List[int], device_arg: str = "auto",
                       debug: bool = False, skip_existing: bool = False,
                       conditions: Optional[List[str]] = None) -> None:
    _apply_config()
    _base.run_experiment_4(dataset_key, seeds, device_arg, debug=debug,
                           skip_existing=skip_existing, conditions=conditions)


def summarize_exp_4_3():
    _apply_config()
    return _base.summarize_exp_4()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_exp_4_3()


if __name__ == "__main__":
    main()
