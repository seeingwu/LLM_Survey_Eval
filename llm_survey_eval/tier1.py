from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from scipy.stats import wasserstein_distance

def _value_counts_prob(series: pd.Series, support: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
    s = series.dropna().astype(int)
    if support is None:
        support = sorted(s.unique().tolist())
    counts = s.value_counts().reindex(support, fill_value=0).astype(float).values
    probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return probs, support

def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(float); q = q.astype(float)
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * np.log2(a / b))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _chi2_and_g_test(counts_h: np.ndarray, counts_l: np.ndarray):
    table = np.vstack([counts_h, counts_l])
    chi2 = chi2_contingency(table, lambda_=None)
    g    = chi2_contingency(table, lambda_="log-likelihood")
    return chi2, g

def _cramers_v(chi2_stat: float, n: int, r: int, c: int) -> float:
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return np.nan
    return np.sqrt(chi2_stat / denom)

def _expected_min(expected: np.ndarray) -> float:
    return float(np.min(expected)) if expected.size else np.nan

def run_tier1_comparisons(
    survey_csv: Path,
    llm_csv: Path,
    ordered_features: List[str],
    multinomial_features: List[str],
    continuous_features: List[str],
    mapping_dict: Optional[Dict[str, Dict[str, int]]] = None,
    type_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    id_col: str = "agent_id",
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    mapping_dict = mapping_dict or {}
    type_mappings = type_mappings or {}

    survey_df = pd.read_csv(survey_csv)
    llm_df    = pd.read_csv(llm_csv)

    if id_col in survey_df.columns and id_col in llm_df.columns:
        survey_df = survey_df.dropna(subset=[id_col])
        llm_df    = llm_df.dropna(subset=[id_col])
        merged = survey_df[[id_col] + ordered_features + multinomial_features + continuous_features].merge(
            llm_df[[id_col] + ordered_features + multinomial_features + continuous_features],
            on=id_col, suffixes=("_s", "_l"), how="inner"
        )
    else:
        merged = pd.DataFrame()
        survey_df = survey_df.rename(columns={c: f"{c}_s" for c in survey_df.columns if c in ordered_features + multinomial_features + continuous_features})
        llm_df    = llm_df.rename(columns={c: f"{c}_l" for c in llm_df.columns if c in ordered_features + multinomial_features + continuous_features})

    rows = []

    for col in multinomial_features + ordered_features:
        col_s = f"{col}_s"
        col_l = f"{col}_l"
        if not merged.empty:
            s_series = merged[col_s]
            l_series = merged[col_l]
        else:
            s_series = survey_df.get(col_s, pd.Series(dtype=float))
            l_series = llm_df.get(col_l, pd.Series(dtype=float))
        s_nonan = s_series.dropna().astype(int)
        l_nonan = l_series.dropna().astype(int)
        support = sorted(set(s_nonan.unique()).union(set(l_nonan.unique())))
        p_s, _ = _value_counts_prob(s_series, support)
        p_l, _ = _value_counts_prob(l_series, support)
        counts_s = np.array([(s_nonan == k).sum() for k in support], dtype=float)
        counts_l = np.array([(l_nonan == k).sum() for k in support], dtype=float)
        tv = _total_variation(p_s, p_l)
        js = _js_divergence(p_s, p_l) if (p_s.sum() > 0 and p_l.sum() > 0) else np.nan
        chi2, g = _chi2_and_g_test(counts_s, counts_l)
        chi2_stat, chi2_p, chi2_df, chi2_exp = chi2[0], chi2[1], chi2[2], chi2[3]
        g_stat, g_p, g_df, g_exp = g[0], g[1], g[2], g[3]
        n_total = int(counts_s.sum() + counts_l.sum())
        cramer_v = _cramers_v(chi2_stat, n_total, r=2, c=len(support))
        exp_min  = _expected_min(chi2_exp)
        base = {
            "feature": col,
            "type": "ordinal" if col in ordered_features else "nominal",
            "n_survey": int(counts_s.sum()),
            "n_llm": int(counts_l.sum()),
            "support_k": len(support),
            "TV": tv,
            "JS": js,
            "chi2_stat": chi2_stat,
            "chi2_df": chi2_df,
            "chi2_p": chi2_p,
            "cramers_v": cramer_v,
            "g_stat": g_stat,
            "g_df": g_df,
            "g_p": g_p,
            "expected_min": exp_min,
        }
        if col in ordered_features:
            w1 = np.nan
            if len(s_nonan) > 0 and len(l_nonan) > 0:
                w1 = wasserstein_distance(s_nonan.values.astype(float), l_nonan.values.astype(float))
            mean_s = float(np.mean(s_nonan)) if len(s_nonan) else np.nan
            mean_l = float(np.mean(l_nonan)) if len(l_nonan) else np.nan
            var_s  = float(np.var(s_nonan, ddof=1)) if len(s_nonan) > 1 else np.nan
            var_l  = float(np.var(l_nonan, ddof=1)) if len(l_nonan) > 1 else np.nan
            mean_diff_ratio = (mean_s - mean_l)/mean_s if np.all(np.isfinite([mean_s, mean_l])) else np.nan
            var_ratio = (var_l / var_s) if (np.isfinite(var_s) and var_s > 0 and np.isfinite(var_l)) else np.nan
            base.update({
                "W1": w1,
                "mean_s": mean_s, "mean_l": mean_l, "mean_diff_ratio": mean_diff_ratio,
                "var_s": var_s, "var_l": var_l, "var_ratio_l_over_s": var_ratio,
            })
        rows.append(base)

    out = pd.DataFrame(rows)
    return out
