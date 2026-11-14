# Phase 4 · W9–W10 audit notes and provisional cross-check

**Status:** Software implementation validated on deterministic fixtures; cross-check values below were calculated on the **unchanged analytical columns in the tracked legacy CSV**, which contains 250,000 rows. To exercise schema validation without actual canonical Phase 2 CSV in this ZIP, the offline check used a *temporary, non-deliverable stand-in* `Returns` missingness mask. **Returns is not a feature or a tested outcome in W9–W10.** Therefore its missingness mask does not change these W9–W10 estimates, but only a Windows run on real `ecommerce_customer_clean.csv` can be the formal acceptance run. The stand-in file is not included in this patch.

## W9 checks

| Primary comparison | Preliminary p | Holm p (family of 3) | Interpretation |
|---|---:|---:|---|
| H1: customer Churn vs Gender | 0.1200 | 0.2400 | Do not reject H0 |
| H2: per-customer mean recorded amount vs Churn | 0.6238 | 0.6238 | Do not reject H0 |
| H3: four category means, customer-clustered Wald test | 0.0558 | 0.1674 | Do not reject H0 |

H3 uses customer-clustered sandwich SE and a 3-df omnibus Wald test, not ordinary row-wise ANOVA as a primary significance test. Six pairwise comparisons are exploratory with Holm p-values and Bonferroni simultaneous CIs. The chart no longer uses a cropped-axis bar chart for tiny amount differences. Non-significance does not prove equality.

## W10 checks

| Model | Preliminary test MAE | Preliminary test R² |
|---|---:|---:|
| Mean baseline | 1,251.401 | ≈0 |
| OLS | 1,250.110 | 0.00322 |
| Ridge | 1,250.113 | 0.00322 |
| Lasso | 1,250.145 | 0.00319 |

OLS improves baseline MAE by approximately **1.292 units (0.103% of baseline MAE)**. Illustrative customer-cluster bootstrap 95% CI for improvement is approximately **[0.635, 1.947]** units; practical usefulness is still negligible. Post-purchase Returns and undefined-time Churn are excluded from regression predictors. Neither statistical significance nor a small numeric improvement proves that the model can predict useful real-business revenue.

## Unresolved from Phase 1

- Dataset remains synthetic; monetary unit unspecified.
- `Total Purchase Amount` is different from `Product Price × Quantity` in 249,952 rows; real revenue semantics are **not established**.
- No causal or population-level interpretation is appropriate.
- Historical data have no randomized assignment; the A/B proposal is **not an executed experiment**.

**To finalize:** regenerate all W9/W10 reports and PNGs with `python -m src.run_phase4` on the genuine Phase 2 cleaned CSV, verify manifest SHA-256, and inspect report tables. This provisional cross-check is deliberately separate from the canonical results produced by the runner.
