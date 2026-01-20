import pandas as pd
from pathlib import Path
from .tier1 import run_tier1_comparisons
from .tier2 import compare_assoc
from .tier3 import compute_global_metrics
from .tier4 import evaluate_tier4, summarize_tier4

def evaluate_all_tiers(human_df, llm_df, feature_schema, ordered_features, nominal_features):
    """Systematically evaluate human vs LLM survey data using Tiers 1–4.

    Tier 1 – Descriptive similarity per feature.
    Tier 2 – Global association structure (MAE, RMSE, MaxAbs).
    Tier 3 – Global multivariate fidelity (Energy, MMD, AUC).
    Tier 4 – Inferential equivalence per outcome (DCR, SMR).

    Parameters
    ----------
    human_df : pd.DataFrame
        Human survey data.
    llm_df : pd.DataFrame
        LLM-generated survey data.
    feature_schema : dict
        Schema defining predictors for Tier 4 models.
    ordered_features : list
        List of ordered (ordinal) outcomes.
    nominal_features : list
        List of nominal (categorical) outcomes.

    Returns
    -------
    pd.DataFrame
        Unified summary table of metrics across all four tiers.
    """

    # Tier 1 — descriptive similarity (feature-level)
    t1_rows = []
    for feat in ordered_features + nominal_features:
        s_nonan = human_df[feat].dropna().astype(int)
        l_nonan = llm_df[feat].dropna().astype(int)
        support = sorted(set(s_nonan.unique()) | set(l_nonan.unique()))
        from scipy.stats import wasserstein_distance
        from numpy import mean, var
        from .tier1 import _value_counts_prob, _total_variation
        p_s, _ = _value_counts_prob(s_nonan, support)
        p_l, _ = _value_counts_prob(l_nonan, support)
        tv = _total_variation(p_s, p_l)
        w1 = wasserstein_distance(s_nonan, l_nonan) if feat in ordered_features else None
        mean_s, mean_l = mean(s_nonan), mean(l_nonan)
        var_s, var_l = var(s_nonan, ddof=1), var(l_nonan, ddof=1)
        mean_diff_ratio = (mean_s - mean_l) / mean_s if mean_s != 0 else None
        var_ratio = var_l / var_s if var_s != 0 else None
        t1_rows.append({
            'tier': 'Tier 1', 'variable': feat, 'TV': tv, 'W1': w1,
            'MeanDiff': mean_diff_ratio, 'VarRatio': var_ratio
        })
    tier1_df = pd.DataFrame(t1_rows)

    # Tier 2 — global association matrix comparison
    _, _, _, t2_summary = compare_assoc(human_df, llm_df, ordered_features, nominal_features)
    tier2_df = pd.DataFrame([{ 'tier': 'Tier 2', 'variable': 'GLOBAL',
                               'MAE': t2_summary.mae, 'RMSE': t2_summary.rmse,
                               'MaxAbs': t2_summary.max_abs }])

    # Tier 3 — global multivariate metrics
    t3_metrics = compute_global_metrics(human_df, llm_df, ordered_features, nominal_features)
    tier3_df = pd.DataFrame([{ 'tier': 'Tier 3', 'variable': 'GLOBAL', **t3_metrics }])

    # Tier 4 — inferential equivalence
    outcomes = {f: {'type': 'ordered'} for f in ordered_features}
    outcomes.update({f: {'type': 'multinomial'} for f in nominal_features})
    t4_eval = evaluate_tier4(human_df, llm_df, feature_schema, outcomes)
    tier4_df = summarize_tier4(t4_eval)
    tier4_df = tier4_df.rename(columns={'outcome': 'variable'})
    tier4_df['tier'] = 'Tier 4'

    # Combine all tiers
    combined = pd.concat([tier1_df, tier2_df, tier3_df, tier4_df], ignore_index=True)
    return combined


def summarise_tier_report(table: pd.DataFrame) -> str:
    """Generate a readable summary report of Tier 1–4 results.

    Parameters
    ----------
    table : pd.DataFrame
        Output from evaluate_all_tiers()

    Returns
    -------
    str
        A formatted text report summarising the evaluation results.
    """
    report_lines = []
    report_lines.append("LLM vs Human Survey Evaluation Summary\n" + "="*45 + "\n")

    # Tier 1 summary
    t1 = table[table['tier'] == 'Tier 1']
    if not t1.empty:
        avg_tv = t1['TV'].mean()
        avg_w1 = t1['W1'].mean()
        avg_mean_diff = t1['MeanDiff'].abs().mean()
        avg_var_ratio = t1['VarRatio'].mean()
        report_lines.append(f"Tier 1 – Descriptive Similarity:\n")
        report_lines.append(f"  Average Total Variation (TV): {avg_tv:.3f}\n")
        report_lines.append(f"  Average Wasserstein Distance (W1): {avg_w1:.3f}\n")
        report_lines.append(f"  Mean Difference Ratio (|Δμ/μ|): {avg_mean_diff:.3f}\n")
        report_lines.append(f"  Variance Ratio (LLM/Human): {avg_var_ratio:.3f}\n\n")

    # Tier 2 summary
    t2 = table[table['tier'] == 'Tier 2']
    if not t2.empty:
        mae, rmse, maxabs = t2.iloc[0][['MAE','RMSE','MaxAbs']]
        report_lines.append(f"Tier 2 – Behavioural Association Consistency:\n")
        report_lines.append(f"  Mean Absolute Error: {mae:.3f}\n")
        report_lines.append(f"  Root Mean Square Error: {rmse:.3f}\n")
        report_lines.append(f"  Max Absolute Difference: {maxabs:.3f}\n\n")

    # Tier 3 summary
    t3 = table[table['tier'] == 'Tier 3']
    if not t3.empty:
        e, mmd, auc = t3.iloc[0][['energy_distance','mmd_gaussian','c2st_auc']]
        report_lines.append(f"Tier 3 – Multivariate Behavioural Fidelity:\n")
        report_lines.append(f"  Energy Distance: {e:.3f}\n")
        report_lines.append(f"  MMD (Gaussian): {mmd:.3f}\n")
        report_lines.append(f"  Classifier AUC (Separability): {auc:.3f}\n\n")

    # Tier 4 summary
    t4 = table[table['tier'] == 'Tier 4']
    if not t4.empty:
        avg_dcr = t4['DCR'].mean()
        avg_smr = t4['SMR'].mean()
        report_lines.append(f"Tier 4 – Inferential Equivalence:\n")
        report_lines.append(f"  Average Directional Consistency Rate (DCR): {avg_dcr:.3f}\n")
        report_lines.append(f"  Average Significance Matching Rate (SMR): {avg_smr:.3f}\n\n")

    # Overall interpretation
    report_lines.append("Interpretation Summary:\n")
    report_lines.append("  - Tier 1 captures marginal distribution realism; low TV/W1 means high fidelity.\n")
    report_lines.append("  - Tier 2 reflects internal behavioural structure; smaller RMSE indicates closer relational patterns.\n")
    report_lines.append("  - Tier 3 assesses global realism; high AUC (>0.8) implies LLM data are easily separable.\n")
    report_lines.append("  - Tier 4 shows inferential alignment; higher DCR/SMR suggests consistent reasoning between human and LLM models.\n")

    return "".join(report_lines)

# Example usage:
# summary_text = summarise_tier_report(results_table)
# print(summary_text)
