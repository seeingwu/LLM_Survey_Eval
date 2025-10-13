# llm_survey_eval

[![English](https://img.shields.io/badge/lang-English-blue)](./README.md) [![简体中文](https://img.shields.io/badge/语言-简体中文-red)](./readme_zh_cn.md)

A research toolkit for evaluating **LLM‑generated survey data** against human survey data via a **four‑tier framework**. This repository is **GitHub‑first** (clone or ZIP install from source). A PyPI release will follow after Tier‑4 stabilises.

---

## Installation

### Option A — Clone the repository

```bash
git clone https://github.com/<your-username>/llm_survey_eval.git
cd llm_survey_eval
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### Option B — Download ZIP (no Git required)

1. On GitHub, click **Code → Download ZIP** (or use the packaged ZIP we provide).
2. Unzip, then:

   ```bash
   cd llm_survey_eval
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .
   ```
3. (Optional) Run the demo notebook:

   ```bash
   pip install jupyter
   jupyter notebook examples/demo_full_pipeline.ipynb
   ```

> Python ≥ 3.9. Core dependencies: NumPy, pandas, SciPy, scikit‑learn, statsmodels (see `pyproject.toml`).

---

## Data requirements

Prepare **two CSV files**—one for **human** data and one for **LLM** data. Both must satisfy:

1. **Identical feature names and types**

   * Use exactly the same column names and type semantics in both files (e.g., `shopping_frequency` as **ordinal**, `shopping_mode` as **nominal**).
   * In code, declare types via `ordered_features` / `nominal_features` or Tier‑4’s `feature_schema`.

2. **Numerical, not strings**

   * Ordinal: integer‑coded (e.g., 1–5 or 1–6).
   * Nominal: integer category IDs (e.g., 1–7). Internally, Tier‑3/4 use **fixed categories** for one‑hot alignment.
   * Continuous: floats.
   * Binary: 0/1 (if not 0/1, we coerce via a median threshold).

3. **Optional ID alignment**

   * If both CSVs share a key (default `agent_id`), Tier‑1/2 align via inner join; otherwise marginals/associations use all rows per file.

4. **Fixed category sets / levels (recommended)**

   * For all nominal predictors and nominal outcomes, explicitly provide `categories`/`levels` (e.g., `[1,2,3,4,5,6,7]`) so one‑hot dimensions and the reference class are consistent across human/LLM. This prevents coefficient‑name mismatches in Tier‑4.

> If your raw data uses text labels, map them to integers during preprocessing, using the **same mapping** for human and LLM.

---

## Four tiers at a glance

1. **Tier‑1 — Descriptive similarity** (marginals)
   Nominal/ordinal: TV, JS, χ², G‑test, Cramér’s V; ordinal adds W₁ and mean/variance comparisons. Continuous: KS, W₁, mean/variance.
2. **Tier‑2 — Association consistency** (pairwise structure)
   Ordered–ordered: Spearman ρ; nominal–nominal: Cramér’s V; ordered–nominal: correlation ratio η. Outputs Human/LLM/Diff matrices + MAE/RMSE/|max| summary.
3. **Tier‑3 — Joint shape**
   Mixed embedding (ordered → [0,1], nominal → one‑hot with fixed categories), then Energy Distance (√ED²), Gaussian‑kernel MMD (median‑heuristic bandwidth), and C2ST AUC (logistic classifier).
4. **Tier‑4 — Inferential equivalence**
   Ordered outcomes: **Ordered Logit**; nominal outcomes: **Multinomial Logit**. With the human model as reference, compute DCR (Directional Consistency Rate) and SMR (Significance Matching Rate), plus a coefficient‑level alignment table for auditability.

---

## Quick start

Use `examples/demo_full_pipeline.ipynb` (now covering **Tiers 1–4**). It builds reproducible toy human/LLM data, runs all tiers, and saves plots:

* Ordinal overlays and nominal composition plots;
* Tier‑2 triptych and |Δ| heatmap;
* Tier‑3 PCA projection;
* Tier‑4 DCR/SMR summary and coefficient‑level details.

---

## API heads‑up (selected)

* `llm_survey_eval.tier1.run_tier1_comparisons(...) -> DataFrame`
* `llm_survey_eval.tier2.tier2_structural(...) -> dict` and `plot_three(...) -> Figure`
* `llm_survey_eval.tier3_alt.compute_global_metrics(...) -> dict`
* `llm_survey_eval.tier4.evaluate_tier4(...) -> dict` and `summarize_tier4(...) -> DataFrame`

---

## Current limitations & roadmap

* The project is **GitHub‑first**; PyPI release will follow once Tier‑4 settles.
* Visualisations are research‑grade; we plan **richer figures** (radar charts, coefficient forests, interactive matrices).
* Metric options will expand (alternative energy‑distance conventions, regularised/robust regressions, feature‑level DCR/SMR aggregation, etc.).
* Robustness for extreme sparsity/small‑N is a work‑in‑progress; consider merging rare categories and checking expected counts.

---

## Licence & citation

MIT Licence (see `LICENSE`).

> Wu, S. (2025). *Evaluating LLM‑Generated Survey Data: A Four‑Tier Framework for Behavioural Equivalence.* University of Leeds.
> Software: `llm_survey_eval`, v0.1.0.
