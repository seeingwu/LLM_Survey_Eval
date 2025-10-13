# llm_survey_eval

Four‑tier evaluation of **LLM‑generated survey data** against **human surveys**.

## Installation
```bash
pip install llm_survey_eval
```

## Quick start
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
print(res2["summary"])

# Tier‑3 (alt)
nominal_categories = {c: [1,2,3,4,5,6,7] for c in nominal}
res3 = compute_global_metrics(human, llm, ordered, nominal,
                              nominal_categories=nominal_categories,
                              seed=42, verbose=False)
print(res3)
```

## Licence
MIT
