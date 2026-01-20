from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)

def _onehot_fixed(s: pd.Series, categories: list[int]) -> np.ndarray:
    c = pd.Categorical(s, categories=categories)
    k = len(categories)
    o = np.zeros((len(c), k), float)
    idx = c.codes
    m = idx >= 0
    o[np.arange(len(c))[m], idx[m]] = 1.0
    return o

def _embed(df: pd.DataFrame, ordered: list[str], nominal: list[str],
           nominal_categories: dict[str, list[int]] | None = None) -> np.ndarray:
    parts = []
    for c in ordered:
        v = np.asarray(df[c].to_numpy())
        if v.ndim != 1:
            raise ValueError(f"Ordered column {c} became {v.ndim}D with shape {v.shape}")
        parts.append(_minmax(v.reshape(-1, 1)))

    for c in nominal:
        cats = nominal_categories.get(c) if nominal_categories else sorted(df[c].dropna().unique())
        oh = _onehot_fixed(df[c], list(cats))
        if oh.ndim != 2:
            raise ValueError(f"Onehot for {c} is {oh.ndim}D with shape {oh.shape}")
        parts.append(oh)

    X = np.hstack(parts) if parts else np.zeros((len(df), 0))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    if len(X)==0 or len(Y)==0:
        return np.nan
    d_xx = float(np.mean(cdist(X, X, 'euclidean')))
    d_yy = float(np.mean(cdist(Y, Y, 'euclidean')))
    d_xy = float(np.mean(cdist(X, Y, 'euclidean')))
    e2 = 2*d_xy - d_xx - d_yy
    return float(np.sqrt(max(e2, 0.0)))

def gaussian_mmd(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None, max_samples: int = 4000, seed: int = 0) -> tuple[float,float]:
    rng = np.random.default_rng(seed)
    Xs = X if len(X)<=max_samples else X[rng.choice(len(X), max_samples, replace=False)]
    Ys = Y if len(Y)<=max_samples else Y[rng.choice(len(Y), max_samples, replace=False)]
    Z  = np.vstack([Xs, Ys])
    if bandwidth is None:
        D = pdist(Z, 'euclidean')
        med = np.median(D[D>0]) if np.any(D>0) else 1.0
        bandwidth = med if np.isfinite(med) and med>0 else 1.0
    def k(a,b):
        D = cdist(a,b,'euclidean')
        return np.exp(-(D**2)/(2*bandwidth**2))
    Kxx, Kyy, Kxy = k(Xs,Xs), k(Ys,Ys), k(Xs,Ys)
    n, m = len(Xs), len(Ys)
    mmd2 = (Kxx.sum()-np.trace(Kxx))/(n*(n-1)) + (Kyy.sum()-np.trace(Kyy))/(m*(m-1)) - 2*Kxy.mean()
    return float(np.sqrt(max(mmd2,0.0))), float(bandwidth)

def c2st_auc_logreg(X: np.ndarray, Y: np.ndarray, test_size: float=0.3, seed: int=0, C: float=1.0, max_iter: int=2000) -> float:
    X_all = np.vstack([X, Y])
    y_all = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=test_size, stratify=y_all, random_state=seed)
    clf = LogisticRegression(max_iter=max_iter, C=C, solver='lbfgs')
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    return float(auc)

def compute_global_metrics(human_df: pd.DataFrame, llm_df: pd.DataFrame,
                           ordered_features: list[str], nominal_features: list[str],
                           mmd_max_samples: int = 4000, seed: int = 0,
                           nominal_categories: dict[str, list[int]] | None = None,
                           verbose: bool = True) -> dict[str, float]:
    if nominal_categories is None:
        nominal_categories = {
            c: sorted(set(human_df[c].dropna().unique()) | set(llm_df[c].dropna().unique()))
            for c in nominal_features
        }
    Xh = _embed(human_df, ordered_features, nominal_features, nominal_categories)
    Xl = _embed(llm_df,   ordered_features, nominal_features, nominal_categories)
    if verbose:
        print("Human embedding shape:", Xh.shape)
        print("LLM embedding shape:", Xl.shape)
    e = energy_distance(Xh, Xl)
    mmd, bw = gaussian_mmd(Xh, Xl, bandwidth=None, max_samples=mmd_max_samples, seed=seed)
    auc = c2st_auc_logreg(Xh, Xl, seed=seed)
    return {"energy_distance": e, "mmd_gaussian": mmd, "mmd_bandwidth": bw, "c2st_auc": auc}
