# llm_survey_eval

A research toolkit for evaluating **LLM‑generated survey data** against human survey data via a **4‑tier framework**. This repository currently targets **GitHub‑first usage** (clone and install from source). A PyPI release will follow after Tier‑4 stabilises.

---

## Why this project

Large Language Models can generate synthetic microdata that superficially resemble human surveys. The central methodological question is: *to what extent are the synthetic distributions, structures, and inferences equivalent to those derived from real survey data?* This toolkit provides reproducible baselines across four tiers to quantify similarity and highlight failure modes.

We aim for transparent, testable, and academically reproducible code. Metrics are standard in statistics and computational social science; where conventions differ (e.g., energy distance with/without square root), we document the chosen convention explicitly.

---

## Installation (GitHub version)

```bash
# 1) clone
git clone https://github.com/<your-username>/llm_survey_eval.git
cd llm_survey_eval

# 2) optional: use a fresh environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) install from source (editable mode)
pip install -e .

# 4) test a quick import
python -c "import llm_survey_eval as m; print(m.__all__)"
```

**Dependencies:** NumPy, pandas, SciPy, scikit‑learn (see `pyproject.toml`). Python ≥ 3.9.

---

## Data expectations

- **Variable coding**
  - *Ordinal* variables must be integer‑coded (e.g., 1–5 or 1–6). These are min–max scaled to [0,1] in Tier‑3.
  - *Nominal* variables must be integer‑coded category IDs (e.g., 1–7). One‑hot encoding uses **fixed category sets** to align dimensions across datasets.
  - *Continuous* variables (Tier‑1 only at present) are floats.
- **ID alignment**: If both CSVs include a respondent key (default `agent_id`), Tier‑1/2 align on inner join for like‑for‑like comparisons; otherwise marginals/associations use all available rows per file.

---

## Four tiers at a glance

1. **Tier‑1: Descriptive similarity (marginals)**
   - Nominal/ordinal: total variation (TV), Jensen–Shannon divergence (JS), χ², G‑test, Cramér’s V; ordinal adds Wasserstein‑1 (W₁), mean/variance comparisons.
   - Continuous: Kolmogorov–Smirnov (KS), W₁, mean/variance.
2. **Tier‑2: Behavioural association consistency (pairwise structure)**
   - Ordered–ordered: Spearman’s ρ;
   - Nominal–nominal: Cramér’s V;
   - Ordered–nominal: correlation ratio η.
   - Outputs Human, LLM, and Difference matrices with summary MAE/RMSE/|max| over the upper triangle.
3. **Tier‑3: Multivariate behavioural fidelity (joint shape)**
   - Mixed embedding (ordered → [0,1], nominal → one‑hot with fixed categories), then:
   - Energy distance (reporting √ED² by default in `tier3_alt`), Gaussian‑kernel MMD (median heuristic bandwidth) with unbiased estimator, C2ST AUC via logistic regression with held‑out test split.
4. **Tier‑4: Inferential equivalence (planned)**
   - Compare model‑based inference between human and LLM data: coefficient signs/magnitudes/SE, predictive calibration. Candidates: logit/probit/ordered models; causal contrasts where appropriate.

---

## Quick start (Tiers 1–3)

```python
import pandas as pd
from llm_survey_eval.tier1 import run_tier1_comparisons
from llm_survey_eval.tier2 import tier2_structural, plot_three
from llm_survey_eval.tier3_alt import compute_global_metrics

human = pd.read_csv("data/sampled_data.csv")
llm   = pd.read_csv("data/dsv3.csv")

# Tier‑1
out1 = run_tier1_comparisons(
    survey_csv="data/sampled_data.csv",
    llm_csv="data/dsv3.csv",
    ordered_features=["shopping_frequency","leisure_frequency","service_frequency",
                      "shopping_satisfaction","leisure_satisfaction","service_satisfaction"],
    multinomial_features=["shopping_mode","leisure_mode","service_mode"],
    continuous_features=[],
    id_col="agent_id",
)
print(out1.head())

# Tier‑2
ordered = ["shopping_frequency","leisure_frequency","service_frequency",
           "shopping_satisfaction","leisure_satisfaction","service_satisfaction"]
nominal = ["shopping_mode","leisure_mode","service_mode"]
res2 = tier2_structural("data/sampled_data.csv","data/dsv3.csv", ordered, nominal, id_col="agent_id")
fig = plot_three(res2["assoc_h"], res2["assoc_l"], res2["assoc_diff"])
fig.savefig("tier2_assoc_triptych.png", dpi=200)
print(res2["summary"])  # mae/rmse/max_abs

# Tier‑3 (alt implementation)
nominal_categories = {c: [1,2,3,4,5,6,7] for c in nominal}  # fix one‑hot alignment
res3 = compute_global_metrics(human, llm, ordered, nominal,
                              nominal_categories=nominal_categories,
                              seed=42, verbose=False)
print(res3)  # energy_distance, mmd_gaussian, mmd_bandwidth, c2st_auc
```

---

## API reference (current modules)

### `llm_survey_eval.tier1`
- `run_tier1_comparisons(survey_csv, llm_csv, ordered_features, multinomial_features, continuous_features, id_col='agent_id', out_csv=None) -> DataFrame`
  - Returns one row per feature with the metrics listed above. If `out_csv` is set, also writes a CSV.

### `llm_survey_eval.tier2`
- `compute_assoc_matrix(df, ordered_cols, nominal_cols) -> DataFrame`
- `compare_assoc(human_df, llm_df, ordered_cols, nominal_cols) -> (A_h, A_l, A_diff, summary)`
- `tier2_structural(survey_csv, llm_csv, ordered_cols, nominal_cols, id_col='agent_id') -> dict`
- `plot_three(assoc_h, assoc_l, assoc_d) -> matplotlib.figure.Figure`

### `llm_survey_eval.tier3_alt`
- `compute_global_metrics(human_df, llm_df, ordered_features, nominal_features, mmd_max_samples=4000, seed=0, nominal_categories=None, verbose=True) -> dict`
- Also exposes `energy_distance`, `gaussian_mmd` (returns value and bandwidth), `c2st_auc_logreg`.

---

## Methodological notes

- **Category alignment matters.** Always pass a fixed `nominal_categories` dict in Tier‑3 to keep embeddings comparable across datasets and model variants.
- **Energy distance convention.** The `tier3_alt` module reports `sqrt(max(ED², 0))` (non‑negative). If you need the un‑squared `ED²` form per some statistics texts, adapt the code or use the alternative Tier‑3 module variant (planned).
- **C2ST AUC stability.** We use a held‑out test split to avoid optimistic bias. For highly imbalanced samples, consider enabling class balancing or calibrating the classifier.
- **Small‑N behaviour.** χ² family metrics can be unstable with sparse cells; inspect `expected_min` from Tier‑1 and consider merging rare categories.

---

## Reproducibility

- Pin a random seed for Tier‑3 routines (`seed=...`).
- Record the MMD **bandwidth** returned by `gaussian_mmd` for exact replication.
- Keep a changelog (see below) and bump versions when interfaces change.

---

## Troubleshooting

- **Windows/WSL**: If WSL is unavailable or proxied, use PowerShell/Python venv; installing from source works without WSL. See the repository’s wiki notes if networking blocks occur.
- **Import errors**: Ensure `pip install -e .` ran in the active environment; verify `python -c "import llm_survey_eval"`.
- **Matplotlib display**: In headless environments, save figures to file (`fig.savefig(...)`) rather than `plt.show()`.

---

## Roadmap

- Tier‑4: inference‑level equivalence (logit/probit/ordered models; bootstrap CI; coefficient sign/magnitude/SE and predictive equivalence tests).
- Expanded tests for Tier‑1/2, synthetic data generators for regression harness.
- Full documentation site with API pages and examples.

---

## Contributing

Pull requests are welcome. Please add unit tests for new metrics or API surfaces. Black/ruff/flake8 style checks will be added before the first PyPI release.

---

## Licence

MIT Licence. See `LICENSE`.

---

## Citation

If you use this toolkit:

> Wu, S. (2025). *Evaluating LLM‑Generated Survey Data: A Four‑Tier Framework for Behavioural Equivalence.* University of Leeds.  
> Software: `llm_survey_eval`, v0.1.0.

---

## Changelog

- **v0.1.0** (GitHub‑first): Tiers 1–3 implemented; Tier‑3 provides `tier3_alt` with √ED² energy distance, Gaussian MMD (median bandwidth), and C2ST AUC. Tier‑4 placeholder.

