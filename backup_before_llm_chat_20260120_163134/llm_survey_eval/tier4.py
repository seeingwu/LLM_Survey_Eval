"""
Tier‑4: Inferential Equivalence

Compares **human** vs **LLM** regression results on two families of discrete outcomes:
  1) Ordered responses (e.g., frequency / satisfaction) via **Ordered Logit**
  2) Nominal responses (e.g., mode choice) via **Multinomial Logit**

Metrics reported (relative to the HUMAN model as reference):
  - DCR (Directional Consistency Rate): proportion of matched coefficients with the same sign
  - SMR (Significance Matching Rate): proportion of matched coefficients whose significance flag matches

Design notes
-----------
* Mixed‑type predictors are handled by a light preprocessor:
  - binary      -> coerced to {0,1}
  - continuous  -> as‑is
  - ordinal     -> kept numeric (integer codes). You may standardise upstream if needed.
  - nominal     -> one‑hot with **fixed categories** (if provided), `drop='first'` for identifiability
* Coefficient matching is done by **coefficient name** (after one‑hot expansion) to avoid ambiguity.
  You can later aggregate to the base feature level if desired.
* Dependencies: requires `statsmodels>=0.13`.

Example
-------
>>> feature_schema = {
...   'gender': {'type': 'binary'},
...   'income': {'type': 'continuous'},
...   'season': {'type': 'nominal', 'categories': [1,2,3,4]},
...   'shopping_frequency': {'type': 'ordinal'},
... }
>>> outcomes = {
...   'satisfaction': {'type': 'ordered', 'levels': [1,2,3,4,5]},
...   'mode_choice' : {'type': 'multinomial', 'levels': [1,2,3,4,5,6,7]}
... }
>>> evals = evaluate_tier4(human_df, llm_df, feature_schema, outcomes, alpha=0.05)
>>> evals['satisfaction']['metrics']  # {'DCR': ..., 'SMR': ...}
>>> plot_forest_tier4(evals,'mode_out',model_labels=('human', 'llm'))

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels for Ordered Logit & Multinomial Logit
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

import matplotlib.pyplot as plt
import scipy.stats as st
# ------------------------- utilities -------------------------

def _ensure_binary(vec: pd.Series) -> pd.Series:
    v = pd.to_numeric(vec, errors='coerce')
    # Map arbitrary truthy/falsy to {0,1}
    if set(pd.unique(v.dropna())) <= {0, 1}:
        return v.astype(float)
    # common cases like {1,2} or {-1,1} -> shift/clip
    v = v.fillna(0)
    v = (v > v.median()).astype(float)
    return v


def _one_hot(df: pd.DataFrame, name: str, categories: Optional[List] = None) -> pd.DataFrame:
    s = df[name]
    if categories is not None:
        s = pd.Categorical(s, categories=categories)
    d = pd.get_dummies(s, prefix=name, drop_first=True)
    return d


def build_design_matrix(df: pd.DataFrame, feature_schema: Dict[str, Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """Build an exogenous design matrix X from a mixed‑type feature schema.

    Schema format per feature:
      { 'type': 'binary' | 'continuous' | 'ordinal' | 'nominal',
        'categories': [... optional for nominal ...] }
    Returns (X, colnames). Intercept is NOT added here.
    """
    parts = []
    names = []

    for feat, spec in feature_schema.items():
        ftype = spec.get('type', 'continuous').lower()
        if feat not in df.columns:
            raise KeyError(f"build_design_matrix: missing feature '{feat}' in dataframe (columns: {list(df.columns)})")

        if ftype == 'binary':
            col = _ensure_binary(df[feat]).astype(float)
            parts.append(col.to_frame(feat))
            names.append(feat)

        elif ftype == 'continuous':
            col = pd.to_numeric(df[feat], errors='coerce')
            parts.append(col.to_frame(feat))
            names.append(feat)

        elif ftype == 'ordinal':
            # keep numeric integer codes
            col = pd.to_numeric(df[feat], errors='coerce')
            parts.append(col.to_frame(feat))
            names.append(feat)

        elif ftype == 'nominal':
            cats = spec.get('categories')
            d = _one_hot(df, feat, categories=cats)
            parts.append(d)
            names.extend(list(d.columns))

        else:
            raise ValueError(f"Unknown feature type '{ftype}' for feature '{feat}'")

    if not parts:
        raise ValueError("No features provided to build_design_matrix.")

    X = pd.concat(parts, axis=1)

    # Drop columns with all-NaN or zero variance to aid identifiability
    nunique = X.nunique(dropna=True)
    keep = nunique[(nunique > 1)].index
    X = X[keep]

    X = X.apply(pd.to_numeric, errors='coerce').astype(float)

    return X, list(X.columns)


# --------------------- model fitting helpers ---------------------

@dataclass
class ModelResult:
    table: pd.DataFrame  # index = coefficient names; columns = ['coef','pval','sign'] (and possibly 'alt')
    model_type: str      # 'ordered' or 'multinomial'
    meta: Dict           # any extra info (levels, reference, etc.)


def _ordered_logit(y: pd.Series, X: pd.DataFrame) -> ModelResult:
    # y must be ordered categorical with increasing levels
    levels = sorted(pd.unique(y.dropna()))
    endog = pd.Categorical(y, categories=levels, ordered=True)

    # Add intercept inside OrderedModel via 'X' with constant if desired; here we exclude explicit intercept
    # to match the latent threshold parameterisation (µ_k play the role of cutpoints/intercepts).
    mod = OrderedModel(endog, X, distr='logit')
    res = mod.fit(method='bfgs', disp=False)

    coefs = res.params.loc[X.columns]  # exclude cutpoints ("thresholds")
    pvals = res.pvalues.loc[X.columns]

    tab = pd.DataFrame({
        'coef': coefs.astype(float),
        'pval': pvals.astype(float),
    })
    tab['sign'] = np.sign(tab['coef']).replace({-1.0: -1, 0.0: 0, 1.0: 1}).astype(int)
    return ModelResult(table=tab, model_type='ordered', meta={'levels': levels})


def _multinomial_logit(y: pd.Series, X: pd.DataFrame) -> ModelResult:
    # y should be categorical with M classes; statsmodels MNLogit uses LAST category as base by default.
    levels = sorted(pd.unique(y.dropna()))
    endog = pd.Categorical(y, categories=levels)
    # Add intercept explicitly
    Xc = sm.add_constant(X, has_constant='add')
    mod = sm.MNLogit(endog.codes, Xc)
    res = mod.fit(method='newton', maxiter=200, disp=False)

    # Params shape: (n_params, M-1), columns correspond to non‑base classes.
    params = res.params  # DataFrame (n_params) x (M-1)
    pvals  = res.pvalues

    rows = []
    for alt_idx, alt in enumerate(params.columns):
        for name in params.index:
            if name == 'const':
                continue  # skip intercept for comparison
            rows.append({
                'coef_name': f"{name}|alt={levels[alt_idx]}",
                'coef': float(params.loc[name, alt]),
                'pval': float(pvals.loc[name, alt]),
            })
    tab = pd.DataFrame(rows).set_index('coef_name')
    tab['sign'] = np.sign(tab['coef']).replace({-1.0: -1, 0.0: 0, 1.0: 1}).astype(int)

    return ModelResult(table=tab, model_type='multinomial', meta={'levels': levels, 'base': levels[-1]})


# --------------------- public fit & compare ---------------------

def fit_discrete_model(df: pd.DataFrame, outcome: str, outcome_type: str, feature_schema: Dict[str, Dict]) -> ModelResult:
    """Fit an ordered or multinomial logit depending on outcome_type.

    outcome_type in {'ordered','multinomial'}; for 'ordered', ensure outcome coded with increasing ints.
    """
    if outcome not in df.columns:
        raise KeyError(f"Outcome '{outcome}' not found in dataframe (columns: {list(df.columns)})")

    X, colnames = build_design_matrix(df, feature_schema)

    X = X.apply(lambda c: pd.to_numeric(c, errors='coerce')).astype(float)

    if outcome_type == 'ordered':
        y = pd.to_numeric(df[outcome], errors='coerce')
        mask = (~y.isna()) & (~X.isna().any(axis=1))
        y = y.loc[mask]
        X = X.loc[mask]
        nunique = X.nunique(dropna=True)
        X = X.loc[:, nunique[nunique > 1].index]
        if X.shape[1] == 0:
            raise ValueError("No non-constant predictors after filtering.")
        return _ordered_logit(y, X)

    elif outcome_type == 'multinomial':
        y = df[outcome]
        mask = (~pd.isna(y)) & (~X.isna().any(axis=1))
        y = y.loc[mask]
        X = X.loc[mask]
        nunique = X.nunique(dropna=True)
        X = X.loc[:, nunique[nunique > 1].index]
        if X.shape[1] == 0:
            raise ValueError("No non-constant predictors after filtering.")
        return _multinomial_logit(y, X)
    
    else:
        raise ValueError("outcome_type must be 'ordered' or 'multinomial'")


def compare_inference(human: ModelResult, llm: ModelResult, alpha: float = 0.05) -> Dict[str, object]:
    """Compute DCR/SMR by matching coefficient names.

    Returns dict with metrics and a joined table for inspection.
    """
    if human.model_type != llm.model_type:
        raise ValueError(f"Model types differ: human={human.model_type}, llm={llm.model_type}")

    th = human.table.copy()
    tl = llm.table.copy()

    # align by coefficient index
    common = th.index.intersection(tl.index)
    if len(common) == 0:
        raise ValueError("No overlapping coefficients to compare (check feature encoding).")

    th = th.loc[common]
    tl = tl.loc[common]

    sig_h = (th['pval'] < alpha).astype(int)
    sig_l = (tl['pval'] < alpha).astype(int)

    # DCR: same sign
    dcr = float(np.mean((np.sign(th['coef']) == np.sign(tl['coef'])).astype(int)))
    # SMR: same significance flag
    smr = float(np.mean((sig_h == sig_l).astype(int)))

    joined = pd.DataFrame({
        'coef_h': th['coef'], 'pval_h': th['pval'], 'sign_h': np.sign(th['coef']).astype(int),
        'coef_l': tl['coef'], 'pval_l': tl['pval'], 'sign_l': np.sign(tl['coef']).astype(int),
        'match_sign': (np.sign(th['coef']) == np.sign(tl['coef'])).astype(int),
        'match_sig' : (sig_h == sig_l).astype(int),
    })

    return {'DCR': dcr, 'SMR': smr, 'n_coefs': int(len(common)), 'detail': joined}


def evaluate_tier4(human_df: pd.DataFrame,
                   llm_df: pd.DataFrame,
                   feature_schema: Dict[str, Dict],
                   outcomes: Dict[str, Dict],
                   alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """Run Tier‑4 comparison across multiple outcomes.

    Parameters
    ----------
    human_df, llm_df : DataFrames containing the outcome and predictor columns.
    feature_schema   : dict mapping predictor name -> {'type': ..., 'categories': ...?}
    outcomes         : dict mapping outcome_name -> {'type': 'ordered'|'multinomial', 'levels': [... optional ...]}
    alpha            : significance threshold for SMR.

    Returns
    -------
    dict outcome_name -> {'metrics': {'DCR': .., 'SMR': .., 'n_coefs': ..},
                          'human': ModelResult,
                          'llm'  : ModelResult}
    """
    results = {}
    for out_name, spec in outcomes.items():
        otype = spec.get('type')
        if otype not in {'ordered', 'multinomial'}:
            raise ValueError(f"Outcome '{out_name}' must set type in {{'ordered','multinomial'}}")

        # Optionally coerce levels (ensures consistent coding across frames)
        if 'levels' in spec and spec['levels'] is not None:
            levels = list(spec['levels'])
            if otype == 'ordered':
                human_df[out_name] = pd.Categorical(human_df[out_name], categories=levels, ordered=True)
                llm_df[out_name]   = pd.Categorical(llm_df[out_name],   categories=levels, ordered=True)
            else:  # multinomial
                human_df[out_name] = pd.Categorical(human_df[out_name], categories=levels)
                llm_df[out_name]   = pd.Categorical(llm_df[out_name],   categories=levels)

        h_mod = fit_discrete_model(human_df, out_name, otype, feature_schema)
        l_mod = fit_discrete_model(llm_df,   out_name, otype, feature_schema)

        metrics = compare_inference(h_mod, l_mod, alpha=alpha)
        results[out_name] = {'metrics': {'DCR': metrics['DCR'], 'SMR': metrics['SMR'], 'n_coefs': metrics['n_coefs']},
                             'human': h_mod, 'llm': l_mod, 'detail': metrics['detail']}
    return results


# --------------------- convenience pretty print ---------------------

def summarize_tier4(evals: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Collect per‑outcome DCR/SMR into a tidy table."""
    rows = []
    for out_name, obj in evals.items():
        m = obj['metrics']
        rows.append({'outcome': out_name, 'DCR': m['DCR'], 'SMR': m['SMR'], 'n_coefs': m['n_coefs']})
    return pd.DataFrame(rows).sort_values('outcome').reset_index(drop=True)

def plot_forest_tier4(result, outcome_name, model_labels=('human', 'llm'), alpha=0.05, figsize=(8, 10)):
    """Plot a forest plot for Tier‑4 results showing coefficient comparison between human and LLM models.

    If standard errors are unavailable, 95% confidence intervals are approximated from p‑values
    using the normal quantile of the two‑tailed test.
    """
    if outcome_name not in result:
        raise ValueError(f"Outcome '{outcome_name}' not found in provided Tier‑4 results.")

    human_tab = result[outcome_name]['human'].table.copy()
    llm_tab = result[outcome_name]['llm'].table.copy()

    common = human_tab.index.intersection(llm_tab.index)
    human_tab = human_tab.loc[common]
    llm_tab = llm_tab.loc[common]

    df = pd.DataFrame({
        'coef_h': human_tab['coef'],
        'coef_l': llm_tab['coef'],
        'pval_h': human_tab['pval'],
        'pval_l': llm_tab['pval']
    }, index=common)

    # approximate stderr from p‑values and coefficients (two‑tailed normal)
    z_h = np.abs(st.norm.ppf(df['pval_h'] / 2))
    z_l = np.abs(st.norm.ppf(df['pval_l'] / 2))

    df['stderr_h'] = np.where(z_h > 0, np.abs(df['coef_h'] / z_h), np.nan)
    df['stderr_l'] = np.where(z_l > 0, np.abs(df['coef_l'] / z_l), np.nan)

    zcrit = st.norm.ppf(1 - alpha / 2)
    df['ci_low_h'] = df['coef_h'] - zcrit * df['stderr_h']
    df['ci_high_h'] = df['coef_h'] + zcrit * df['stderr_h']
    df['ci_low_l'] = df['coef_l'] - zcrit * df['stderr_l']
    df['ci_high_l'] = df['coef_l'] + zcrit * df['stderr_l']

    df = df.reindex(df['coef_h'].abs().sort_values(ascending=False).index)

    y_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=figsize)

    ax.hlines(y_pos + 0.2, df['ci_low_h'], df['ci_high_h'], color='tab:blue', lw=2)
    ax.hlines(y_pos - 0.2, df['ci_low_l'], df['ci_high_l'], color='tab:orange', lw=2)

    ax.plot(df['coef_h'], y_pos + 0.2, 'o', color='tab:blue', label=model_labels[0])
    ax.plot(df['coef_l'], y_pos - 0.2, 's', color='tab:orange', label=model_labels[1])

    ax.axvline(x=0, color='grey', linestyle='--', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()

    ax.set_xlabel('Coefficient Estimate')
    ax.set_title(f'Forest Plot: {outcome_name} ({model_labels[0]} vs {model_labels[1]})')
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    return fig, ax
