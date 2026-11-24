"""Exp4.2: low-alpha continuous spectral OneTrans (all alpha nodes <= 1).

Reuses the exp4 continuous-spectral machinery by swapping the experiment
config on the shared module before dispatching. Importing this module is
side-effect free; the config is applied only when the runner entry points
below are called.
"""

from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.exp_4_2_low_alpha import CONFIG as EXP_4_2
import src.experiments.continuous_spectral as _base


def _apply_config() -> None:
    _base.EXP_4 = EXP_4_2
    _base.EXP_ID = EXP_4_2["experiment"]
    _base.EXP4_ALPHAS = [float(a) for a in EXP_4_2["spectral"]["alpha_nodes"]]


def run_experiment_4_2(dataset_key: str, seeds: List[int], device_arg: str = "auto",
                       debug: bool = False, skip_existing: bool = False,
                       conditions: Optional[List[str]] = None) -> None:
    _apply_config()
    _base.run_experiment_4(dataset_key, seeds, device_arg, debug=debug,
                           skip_existing=skip_existing, conditions=conditions)


def summarize_exp_4_2():
    _apply_config()
    return _base.summarize_exp_4()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_exp_4_2()


if __name__ == "__main__":
    main()
