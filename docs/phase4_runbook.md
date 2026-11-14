# Phase 4: Statistical Inference & Regression · W9–W10

Baseline: `Dung` commit `ec406de` (Phase 3). Scope: #15 (hypotheses), #16 (A/B test design), #17 (regression). **Keep W8 and all EX exercises out of scope.** Use `docs/data_dictionary.md` for semantic limits.

## What changed

- W9 code now reads only the **Phase 2 validated** `data/processed/ecommerce_customer_clean.csv`, not tracked legacy CSV. H1/H2 work at *customer level*. H3 uses cluster-sandwich covariance by Customer ID; each of six post-hoc contrasts is accompanied by adjusted p-value and family-wise CI. Three top-level p-values undergo Holm correction. Hypotheses are **exploratory**, not preregistered.
- W9 plot 09 uses a zero-based churn bar chart and dot+CI for amount differences (no cut-off bars implying huge effects).
- W10 target remains `Total Purchase Amount` but is explicitly an **unverified recorded field**. Excludes post-purchase `Returns` and undefined Churn; no automatic null-to-zero. A grouped 80/20 holdout and grouped 3-fold CV, baseline/OLS/Ridge/Lasso, customer-clustered OLS coefficient CIs, a customer-bootstrap CI for the MAE gain, and diagnostics are generated.
- Removed misleading Durbin-Watson interpretation from rows sorted across unrelated customers. Cluster SE does not establish causality or correct the unresolved target semantics.
- A/B test is a **future protocol only** (`docs/ab_test_design.md`); source has no treatment assignment or exposure logs.
- Runner `python -m src.run_phase4` records source/artifact SHA-256 in a local manifest, and fails if the clean CSV is absent or invalid. No changes to Phase 1–3 datasets.

## Windows / Python 3.11 execution

```powershell
cd C:\Code\KHDL\Project

git switch Dung
git status
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.run_phase4 --dry-run
.\venv\Scripts\python.exe -m src.run_phase4

git status
```

Or individually: `python -m src.run_phase4 --only w9` and `python -m src.run_phase4 --only w10` (the same validated input). Direct entrypoints `python -m src.pipeline.07_hypothesis_tests` and `python -m src.pipeline.06_regression` still work. No raw Kaggle data in Git; rebuild clean CSV with `python -m src.run_pipeline` if missing.

## Artifacts

`reports/phase4/phase4_review.md` contains a **provisional** full-row analytical-field cross-check, deliberately distinguished from canonical reports regenerated on Windows.


- `reports/hypothesis_tests.md`, `reports/regression_results.md` (generated findings)
- `reports/figures/09_hypothesis_tests.png`, `06_residual_diagnostics.png`, `07_coefficients_ci.png`, `08_actual_vs_predicted.png` (generated plots; only 06–09 overwritten)
- `reports/phase4/hypothesis_results.json`, `reports/phase4/regression_metrics.json` (machine-readable summaries)
- `reports/phase4/run_manifest.json` (local reproducibility record, ignored by Git)

Generated reports and 06–09 PNGs are **tracked**; rerunning W9/W10 marks them modified in Git, which is expected. Inspect and commit these outputs with Phase 4 after testing on the actual validated clean CSV (not legacy CSV or a test fixture). Do not commit `data/raw`, generated clean CSV, names/identifiers or the local manifest.

## Acceptance gate

1. All Phase 1–4 unit tests PASS; raw→clean integrity gate PASS; full Phase 4 runner reaches PASS.
2. W9 reports customer-level H1/H2, customer-clustered H3, multiplicity correction and no oversized visually misleading bars.
3. W10 has grouped train/test disjointness, training-only preprocessing and grouped CV, OLS vs mean baseline, cluster SE and honest low-signal assessment.
4. Neither W9 nor W10 uses stale `ecommerce_customer_cleaned.csv`. No claim that monetary unit is USD/VND or that recorded amount necessarily equals revenue.
5. A/B plan clearly labeled **design**, not an executed experiment.

**Limitation:** Since the ZIP excludes raw Kaggle CSV and Phase 2 rebuilt clean CSV, complete Phase 4 acceptance must also run on Windows against the validated 250,000-row clean file.
