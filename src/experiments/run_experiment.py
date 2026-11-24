"""Unified experiment entry point for exp 1.1 - 1.4, 2.x, 3 and 4.

Usage: python -m src.experiments.run_experiment --exp 1_1 --dataset 1k
       python -m src.experiments.run_experiment --exp 3 --dataset 1k
       python -m src.experiments.run_experiment --exp 4 --dataset 1k
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import CONFIG as PATHS
from config.dataset import CONFIG as DATA_CFG
from config.onetrans import CONFIG as MODEL_CFG
from config.exp_1_1_count import CONFIG as EXP_1_1
from config.exp_1_2_quality import CONFIG as EXP_1_2
from config.exp_1_3_type import CONFIG as EXP_1_3
from config.exp_1_4_length import CONFIG as EXP_1_4
from config.exp_2_1_task_aware_gate import CONFIG as EXP_2_1
from config.exp_2_2_gate_ablation import CONFIG as EXP_2_2
from config.exp_2_3_gate_analysis import CONFIG as EXP_2_3
from src.data.permutation import build_quality_maps, load_quality_maps
from src.data.dataset import QualityTransform
from src.experiments import figures
from src.experiments.dual_branch import run_dual_experiment
from src.experiments.runner import dist_rank, get_data, run_condition
from src.experiments.static_probe import run_static_probe


def conditions_1_1():
    order = EXP_1_1["field_order"]
    conds = []
    for k in EXP_1_1["feature_counts"]:
        enabled = [f for _, f in order[:k]]
        conds.append((f"K{k}", {"enabled_fields": enabled, "variable": {"K": k}}))
    return conds


def conditions_1_2(dataset_key):
    data = get_data(dataset_key)
    probe_result = None
    if dist_rank() == 0:
        probe_result = run_static_probe(data, dataset_key)
        build_quality_maps(dataset_key, EXP_1_2["quality_levels"], EXP_1_2["permutation_seed"])
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.barrier()  # maps/probe cache must exist before other ranks load them
    conds = []
    for q in EXP_1_2["quality_levels"]:
        transform = None if q >= 1.0 else QualityTransform(load_quality_maps(dataset_key, q), data.tables)
        conds.append((f"q{q:g}", {"transform": transform, "variable": {"q": q},
                                  "enabled_fields": None}))
    return conds, probe_result


def conditions_1_3():
    groups = DATA_CFG["variable_fields"]
    conds = []
    for name, counts in EXP_1_3["conditions"].items():
        enabled = (groups["user"][:counts["user"]] + groups["context"][:counts["context"]]
                   + groups["item"][:counts["item"]])
        conds.append((name, {"enabled_fields": enabled,
                             "variable": {"condition": name,
                                          "K": counts["user"] + counts["context"] + counts["item"]}}))
    return conds


def conditions_1_4():
    conds = []
    for L in EXP_1_4["history_lengths"]:
        conds.append((f"L{L}", {"max_history_len": L, "min_history": EXP_1_4["min_history"],
                                "variable": {"L": L}}))
    return conds


def _gate_overrides(mode, extra=None):
    overrides = {"use_residual_gate": mode != "none"}
    if mode != "none":
        overrides["residual_gate"] = {**EXP_2_1["residual_gate"], "mode": mode, **(extra or {})}
    return overrides


def _shuffle_analyses():
    return {"history_shuffle": {"seed": EXP_2_3["shuffle_seed"]}}


def conditions_2_1():
    return [
        ("Baseline", {"model_overrides": _gate_overrides("none"), "variable": {"group": "Baseline"},
                      "analyses": _shuffle_analyses()}),
        ("TaskAwareGate", {"model_overrides": _gate_overrides("learned"),
                           "variable": {"group": "TaskAwareGate"}, "analyses": _shuffle_analyses()}),
    ]


def conditions_2_2():
    conds = [("NoResidual", {"model_overrides": _gate_overrides("none"),
                             "variable": {"group": "NoResidual"}, "analyses": _shuffle_analyses()}),
             ("Ungated", {"model_overrides": _gate_overrides("ungated"),
                          "variable": {"group": "Ungated"}, "analyses": _shuffle_analyses()}),
             ("LearnedGate", {"model_overrides": _gate_overrides("learned"),
                              "variable": {"group": "LearnedGate"}, "analyses": _shuffle_analyses()})]
    for alpha in EXP_2_2["fixed_gate_alphas"]:
        conds.append((f"Fixed{alpha:g}", {
            "model_overrides": _gate_overrides("fixed", {"fixed_gate_alpha": alpha}),
            "variable": {"group": "FixedGate", "alpha": alpha},
            "analyses": _shuffle_analyses()}))
    return conds


def conditions_2_3():
    ev = EXP_2_3["evaluation"]
    analyses = {}
    if ev.get("compute_history_shuffle"):
        analyses["history_shuffle"] = {"seed": EXP_2_3["shuffle_seed"]}
    if ev.get("compute_gate_statistics"):
        analyses["gate_statistics"] = {"length_buckets": EXP_2_3["history_length_buckets"],
                                       "group_field": EXP_2_3["group_field"]}
    if ev.get("compute_gate_intervention"):
        analyses["gate_intervention"] = True
    return [("LearnedGate", {"model_overrides": _gate_overrides("learned"),
                             "variable": {"group": "LearnedGate"}, "analyses": analyses})]


def _flatten_numbers(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_numbers(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = obj
    return out


def aggregate(all_metrics, conds):
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    analysis_keys = set()
    for m in all_metrics:
        analysis_keys |= set(_flatten_numbers({k: m[k] for k in ("history_shuffle", "gate_intervention")
                                               if k in m}))
        if "gate_stats" in m:
            analysis_keys |= set(_flatten_numbers({"gate_stats": m["gate_stats"]["overall"]}))
    summary = {}
    for cond_name, _ in conds:
        runs = [m for m in all_metrics if m["condition"] == cond_name]
        if not runs:
            continue
        entry = {"n_seeds": len(runs)}
        for k in keys:
            vals = [r[k] for r in runs if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
            if vals:
                entry[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for k in sorted(analysis_keys):
            vals = [_flatten_numbers({kk: r[kk] for kk in ("history_shuffle", "gate_intervention") if kk in r}
                                     | ({"gate_stats": r["gate_stats"]["overall"]} if "gate_stats" in r else {})
                                     ).get(k) for r in runs]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                entry[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        ner_runs = [r["ner"] for r in runs]
        entry["ner"] = {
            layer: {"mean": float(np.mean([n[layer] for n in ner_runs])),
                    "std": float(np.std([n[layer] for n in ner_runs]))}
            for layer in ner_runs[0]
        }
        entry["seeds"] = [r["seed"] for r in runs]
        summary[cond_name] = entry
    return summary


def run_experiment(exp: str, dataset_key: str, seeds, device_arg: str, debug: bool = False,
                   skip_existing: bool = False):
    exp_id = {"1_1": "exp_1_1_count", "1_2": "exp_1_2_quality",
              "1_3": "exp_1_3_type", "1_4": "exp_1_4_length",
              "2_1": EXP_2_1["experiment"], "2_2": EXP_2_2["experiment"],
              "2_3": EXP_2_3["experiment"]}[exp]
    probe_result = None
    builders = {"1_1": conditions_1_1, "1_3": conditions_1_3, "1_4": conditions_1_4,
                "2_1": conditions_2_1, "2_2": conditions_2_2, "2_3": conditions_2_3}
    if exp == "1_2":
        conds, probe_result = conditions_1_2(dataset_key)
    else:
        conds = builders[exp]()

    all_metrics = []
    if exp == "1_1" and dist_rank() == 0:
        meta_fp = PATHS["metadata_root"] / "field_order.json"
        meta_fp.parent.mkdir(parents=True, exist_ok=True)
        meta_fp.write_text(json.dumps({
            "field_order": [f for _, f in EXP_1_1["field_order"]],
            "field_groups": {g: [f for gg, f in EXP_1_1["field_order"] if gg == g] for g in ("user", "context", "item")},
            "feature_counts": EXP_1_1["feature_counts"],
        }, indent=2))
    for cond_name, kwargs in conds:
        for seed in seeds:
            metrics = run_condition(f"{exp_id}_{cond_name}", cond_name, dataset_key, seed,
                                    enabled_fields=kwargs.get("enabled_fields"),
                                    transform=kwargs.get("transform"),
                                    max_history_len=kwargs.get("max_history_len"),
                                    min_history=kwargs.get("min_history"),
                                    device_arg=device_arg, debug=debug,
                                    skip_existing=skip_existing,
                                    model_overrides=kwargs.get("model_overrides"),
                                    analyses=kwargs.get("analyses"))
            if metrics is not None:
                metrics["variable"] = kwargs["variable"]
                all_metrics.append(metrics)

    if dist_rank() != 0:
        return
    summary = aggregate(all_metrics, conds)
    out = {"experiment": exp_id, "dataset": dataset_key, "seeds": list(seeds),
           "conditions": [c for c, _ in conds], "summary": summary}
    if probe_result is not None:
        out["static_probe"] = probe_result
    metrics_dir = PATHS["metrics_root"] / exp_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    variables = {c: v["variable"] for c, v in conds}
    ner_by_cond = {c: summary[c]["ner"] for c in summary}
    metric_by_cond = {c: {k: summary[c][k] for k in
                          ("probe_valid_auc", "probe_test_auc", "full_test_auc",
                           "full_test_logloss") if k in summary[c]} for c in summary}
    figures.plot_ner_curves(ner_by_cond, exp_id, PATHS["figures_root"] / f"{exp_id}_ner_layers.png")
    if exp == "1_1":
        figures.plot_ner_vs_x(ner_by_cond, [int(c["K"]) for c in variables.values()],
                              "K (number of non-sequential fields)",
                              PATHS["figures_root"] / f"{exp_id}_ner_vs_K.png")
    if exp == "1_2":
        figures.plot_ner_vs_x(ner_by_cond, [float(c["q"]) for c in variables.values()],
                              "q (field quality level)",
                              PATHS["figures_root"] / f"{exp_id}_ner_vs_q.png")
        figures.plot_field_importance(probe_result["Q_j"],
                                      PATHS["figures_root"] / f"{exp_id}_field_importance.png")
    if exp == "1_4":
        figures.plot_ner_vs_x(ner_by_cond, [int(c["L"]) for c in variables.values()],
                              "L (history length)",
                              PATHS["figures_root"] / f"{exp_id}_ner_vs_L.png")
    figures.plot_metric_bars(metric_by_cond, exp_id, PATHS["figures_root"] / f"{exp_id}_metrics.png")
    if exp.startswith("2"):
        delta_keys = [k for k in ("history_shuffle.delta_order_auc", "history_shuffle.delta_order_logloss")
                      if any(k in summary[c] for c in summary)]
        if delta_keys:
            figures.plot_group_bars({c: {k: summary[c][k] for k in delta_keys} for c in summary},
                                    "History-order shuffling sensitivity",
                                    PATHS["figures_root"] / f"{exp_id}_shuffle_delta.png",
                                    ylabel="Δ (original − permuted)")
    if exp == "2_3":
        figures.plot_gate_outputs(out, PATHS["figures_root"], exp_id)
    print(f"[done] {exp_id}: summary -> {metrics_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True,
                        choices=["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "3", "4"])
    parser.add_argument("--dataset", required=True, choices=["1k", "27k"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip (experiment, condition, seed) runs whose metrics JSON already exists")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="exp 3/4: subset of configured conditions")
    parser.add_argument("--sweep", action="store_true",
                        help="exp 3: run the configured lambda sweep")
    parser.add_argument("--ner-lambda", type=float, default=None,
                        help="exp 3: override the high_only/low_only/dual_split lambda")
    args = parser.parse_args()
    if args.exp == "1_4" and not EXP_1_4["enabled"]:
        print("exp 1_4 is disabled in config/exp_1_4_length.py (enabled=False); nothing to do")
        return
    if args.exp == "3":
        from config.exp_3_dual_branch_ner import CONFIG as EXP_3
        seeds = args.seeds or EXP_3["seeds"]
        run_dual_experiment(args.dataset, seeds, args.device, debug=args.debug,
                            skip_existing=args.skip_existing, conditions=args.conditions,
                            sweep=args.sweep, ner_lambda=args.ner_lambda)
        return
    if args.exp == "4":
        from config.exp_4_continuous_spectral import CONFIG as EXP_4
        from src.experiments.continuous_spectral import run_experiment_4
        seeds = args.seeds or EXP_4["seeds"]
        run_experiment_4(args.dataset, seeds, args.device, debug=args.debug,
                         skip_existing=args.skip_existing, conditions=args.conditions)
        return
    seeds = args.seeds or MODEL_CFG["seeds"]
    run_experiment(args.exp, args.dataset, seeds, args.device, debug=args.debug,
                   skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
