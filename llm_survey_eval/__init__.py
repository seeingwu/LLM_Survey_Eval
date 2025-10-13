from .tier1 import run_tier1_comparisons
from .tier2 import (compute_assoc_matrix, compare_assoc, tier2_structural, plot_three)
from .tier3 import (
    compute_global_metrics as compute_global_metrics_alt,
    energy_distance as energy_distance_alt,
    gaussian_mmd as gaussian_mmd_alt,
    c2st_auc_logreg as c2st_auc_alt,
)

__all__ = [
    "run_tier1_comparisons",
    "compute_assoc_matrix", "compare_assoc", "tier2_structural", "plot_three",
    "compute_global_metrics_alt", "energy_distance_alt", "gaussian_mmd_alt", "c2st_auc_alt",
    "evaluate_tier4", "summarize_tier4", "build_design_matrix", "fit_discrete_model"
]
