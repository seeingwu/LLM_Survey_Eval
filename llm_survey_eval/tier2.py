from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr

# -------------------- helpers --------------------

def _cramers_v_from_counts(table: np.ndarray) -> float:
    chi2, p, df, expected = chi2_contingency(table, correction=False)
    n = table.sum()
    if n == 0:
        return np.nan
    r, c = table.shape
    v = np.sqrt((chi2 / n) / (min(r, c) - 1)) if min(r, c) > 1 else np.nan
    return float(v)


def _correlation_ratio_eta(y_ord: pd.Series, x_nom: pd.Series) -> float:
    y = y_ord.dropna().astype(float)
    x = x_nom.reindex(y.index).astype(object)
    if len(y) == 0:
        return np.nan
    groups = [y[x == g] for g in pd.unique(x)]
    n_total = float(len(y))
    if n_total <= 1:
        return np.nan
    grand = y.mean()
    ss_between = sum([len(g) * (g.mean() - grand) ** 2 for g in groups if len(g) > 0])
    ss_total = ((y - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def _spearman_pair(x: pd.Series, y: pd.Series) -> float:
    r = spearmanr(x, y, nan_policy="omit").correlation
    return float(r)


# -------------------- core API --------------------

def compute_assoc_matrix(df: pd.DataFrame, ordered_cols: List[str], nominal_cols: List[str]) -> pd.DataFrame:
    """Compute a mixed-type association matrix with basic sanity checks."""
    cols = ordered_cols + nominal_cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"compute_assoc_matrix: missing columns in input df: {missing}. Available: {list(df.columns)}")

    n = len(cols)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            ci, cj = cols[i], cols[j]
            if i == j:
                A[i, j] = 1.0
                continue
            if (ci in ordered_cols) and (cj in ordered_cols):
                val = _spearman_pair(df[ci], df[cj])
            elif (ci in nominal_cols) and (cj in nominal_cols):
                tab = pd.crosstab(df[ci], df[cj]).values
                val = _cramers_v_from_counts(tab)
            else:
                if ci in ordered_cols and cj in nominal_cols:
                    val = _correlation_ratio_eta(df[ci], df[cj])
                else:
                    val = _correlation_ratio_eta(df[cj], df[ci])
            A[i, j] = A[j, i] = val
    return pd.DataFrame(A, index=cols, columns=cols)


@dataclass
class AssocSummary:
    mae: float
    rmse: float
    max_abs: float


def compare_assoc(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                  ordered_cols: List[str], nominal_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, AssocSummary]:
    Ah = compute_assoc_matrix(human_df, ordered_cols, nominal_cols)
    Al = compute_assoc_matrix(llm_df, ordered_cols, nominal_cols)
    D = Al - Ah
    mask = np.triu(np.ones_like(D.values, dtype=bool), k=1)
    diffs = D.values[mask]
    summary = AssocSummary(
        mae=float(np.nanmean(np.abs(diffs))),
        rmse=float(np.sqrt(np.nanmean(diffs ** 2))),
        max_abs=float(np.nanmax(np.abs(diffs)))
    )
    return Ah, Al, D, summary


def tier2_structural(survey_csv: Path, llm_csv: Path,
                     ordered_cols: List[str], nominal_cols: List[str],
                     id_col: str = "agent_id") -> Dict[str, object]:
    """Read two CSVs, align by id if present, then compute association matrices and diffs.

    Robustness upgrades:
    - Validates presence of all required columns per file with clear error messages.
    - If id alignment is requested but any required column is missing after merge,
      raises with diagnostics (shows available columns and the exact missing set).
    """
    s = pd.read_csv(survey_csv)
    l = pd.read_csv(llm_csv)

    required = [id_col] + ordered_cols + nominal_cols
    missing_s = [c for c in required if c not in s.columns]
    missing_l = [c for c in required if c not in l.columns]
    if missing_s:
        raise KeyError(f"tier2_structural: survey_csv is missing columns {missing_s}. Available: {list(s.columns)}")
    if missing_l:
        raise KeyError(f"tier2_structural: llm_csv is missing columns {missing_l}. Available: {list(l.columns)}")

    if id_col in s.columns and id_col in l.columns:
        use_cols = [id_col] + ordered_cols + nominal_cols
        m = s[use_cols].merge(l[use_cols], on=id_col, suffixes=("_s", "_l"), how="inner")
        if m.empty:
            raise ValueError("tier2_structural: after inner join on id_col, merged frame is empty. Check id overlap.")
        # Rebuild aligned frames with base names
        hs = {c[:-2]: m[c] for c in m.columns if c.endswith("_s")}
        hl = {c[:-2]: m[c] for c in m.columns if c.endswith("_l")}
        s_df = pd.DataFrame(hs)
        l_df = pd.DataFrame(hl)
    else:
        # fall back to independent frames (no id alignment)
        s_df = s[ordered_cols + nominal_cols].copy()
        l_df = l[ordered_cols + nominal_cols].copy()

    # Final guard: ensure all requested columns exist
    for name, df in [("human_df", s_df), ("llm_df", l_df)]:
        missing = [c for c in (ordered_cols + nominal_cols) if c not in df.columns]
        if missing:
            raise KeyError(f"{name} missing columns {missing}. Columns present: {list(df.columns)}")

    Ah, Al, D, summary = compare_assoc(s_df, l_df, ordered_cols, nominal_cols)
    return {"assoc_h": Ah, "assoc_l": Al, "assoc_diff": D, "summary": summary}


# -------------------- viz --------------------

def plot_three(assoc_h: pd.DataFrame, assoc_l: pd.DataFrame, assoc_d: pd.DataFrame):
    labels = list(assoc_h.columns)
    n = len(labels)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mats = [("Human", assoc_h.values), ("LLM", assoc_l.values), ("Difference (LLM − Human)", assoc_d.values)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (title, data) in zip(axes, mats):
        data_masked = np.ma.array(data, mask=mask)
        im = ax.imshow(data_masked, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels)
        for g in [3, 6]:
            ax.axhline(g-0.5, linewidth=1.0, c='black')
            ax.axvline(g-0.5, linewidth=1.0, c='black')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Association (ρ / η / V)")
    fig.tight_layout()
    return fig

