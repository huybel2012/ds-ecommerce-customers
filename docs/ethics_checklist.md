# Phase 6 — Ethics & Responsible Data Science Checklist

Status values are **PASS**, **LIMITATION**, **OPEN**. "PASS" means verified only for the documented synthetic workflow, not any external legal certification.

| W12 item | Status | Evidence / action |
|---|---|---|
| Provenance/source attribution | PASS | Shriyash Jagtap Kaggle dataset URL in README, data dictionary and model card; Faker synthetic according to source description |
| Permission for data redistribution | **OPEN** | License not independently verified on primary Kaggle page. Before public redistribution, inspect current license/attribution/derivatives terms and record evidence + date |
| Data subjects / consent | LIMITATION | Synthetic data, not actual subjects; synthetic-looking names still tracked in legacy files, and real-world reuse needs independent legal/consent review |
| Identifiers / minimization | **OPEN** | `data/processed/ecommerce_customer_cleaned.csv` and `ecommerce_clean.parquet` are already tracked; the former contains `Customer Name` and `Customer ID`. New Phase 6 artifacts contain only aggregates; legacy data removal/replacement needs human approval |
| Missing/quality limitations | PASS | `Returns` 47,596 missing preserved; 249,952 amount discrepancies documented in Phase 1 |
| Subgroup coverage and underrepresented groups | LIMITATION | Gender only Male/Female; Age 18–70; unknown/unrepresented identities not measurable |
| Performance across groups | PASS after local run | Run `python -m src.run_phase6` to populate `reports/phase6/fairness_metrics.json`, suppress low-support groups and include support/intervals |
| Fairness claim / harm prevention | LIMITATION | Model near chance, predicts almost all Churn at Phase 5 threshold; subgroup parity may be vacuous; no customer decisions authorized |
| Churn label temporal validity | **OPEN** | Static labels have no effective timestamp; no prospective churn prediction claim is valid |
| Scalability | PASS for current scope | 250k rows, Pandas/Parquet; future scaling requires measured memory/runtime, not automatic Spark adoption |
| Model card | PASS after local run | `reports/phase6/model_card.md` produced from frozen Phase 5 holdout and fairness study; review content manually |
| AI assistance transparency | PASS after human sign-off | `docs/ai_usage_disclosure.md` drafted from observed workflow; team must verify actual uses, roles and attribution before submitting |

**Release gate:** Do not describe the dataset as unrestricted/CC0 or anonymized, and do not use the classifier for real decisions. The open license and tracked names questions require explicit resolution for wider public distribution. Reproducibility does not grant permission to republish third-party data.
