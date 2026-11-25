import copy
from typing import Any, Dict

from config.exp_4_2_low_alpha import CONFIG as BASE

CONFIG: Dict[str, Any] = copy.deepcopy(BASE)
CONFIG["experiment"] = "exp_4_3"
CONFIG["summary_alias"] = "summary_exp_4_3"

# Keep the same discrete low-alpha grid for fixed-alpha single links and the
# finite-K discrete representation mix. The continuous representation mix
# integrates over the whole log-alpha interval with Gauss-Legendre nodes.
CONFIG["spectral"]["continuous"] = {
    "alpha_min": 0.125,
    "alpha_max": 1.0,
    "num_nodes": 32,
    "quadrature": "gauss_legendre",
    "integration_variable": "log_alpha",
}

CONFIG["conditions"]["continuous_representation_mix"] = {
    "kind": "continuous_representation_mix",
    "fusion": "contextual",
    "node_aux_loss_weight": 0.0,
}
CONFIG["model"]["fusion_modes"] = list(CONFIG["model"]["fusion_modes"]) + [
    "continuous_representation_mix",
]
CONFIG["default_conditions"] = [
    "baseline_frozen_prefix",
    "fixed_alpha_0.125",
    "fixed_alpha_0.1767767",
    "fixed_alpha_0.25",
    "fixed_alpha_0.35355339",
    "fixed_alpha_0.5",
    "fixed_alpha_0.70710678",
    "contextual_representation_mix",
    "continuous_representation_mix",
]
CONFIG["ablation_conditions"] = []
