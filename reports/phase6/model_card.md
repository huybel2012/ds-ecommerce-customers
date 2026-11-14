# Model Card (1-page) — Retrospective Synthetic Churn Classifier

**Version / status:** Phase 6 · research demonstration ONLY · **NOT FOR DEPLOYMENT**
**Model:** Random Forest (Phase 5 frozen hyperparameters; 49,128 eligible synthetic customers).
**Owner / accountable reviewers:** university project team; final human review required.
**Purpose:** illustrate customer-level binary classification and post-hoc subgroup evaluation for W11–W12; NOT forecast future churn, profile real people, set prices, or allocate interventions.

## Data & protocol
- **Source:** Shriyash Jagtap, [Kaggle e-commerce customer dataset](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis); synthetic Faker data (250,000 transaction rows; 49,673 customers). License **not verified from primary Kaggle metadata**.
- **Observation:** features from transactions before 2023-01-01, static `Churn` label has NO verified event/effective timestamp.
- **Inputs:** 19 aggregate behavior features; `Customer ID` used only to split, names/age/gender/Returns/label excluded from predictors. Age and Gender used for evaluation only.
- **Evaluation:** group-disjoint holdout N=9,826, seed=42; training-only OOF F1 chose threshold=0.375; Phase 6 **replays**, not retunes, this decision. Binary confusion matrix must match exactly; any minor AUC/AP differences (<0.005 absolute) are recorded in fairness report.

## Frozen holdout results
- ROC-AUC=0.4949; PR-AUC=0.1965; F1=0.3293; balanced accuracy=0.4999.
- Confusion matrix [TN, FP, FN, TP]=[3, 7886, 1, 1936]. Nearly all customers are predicted Churn; high recall here is NOT evidence of useful discrimination.
- Phase 5.1 added temporal features and ran shuffled-label controls; **no reliable improvement** was found.

## Fairness / uncertainty
- Post-hoc Gender, AgeBand and Gender×Age subgroup metrics include selection rate, TPR, FPR, PPV, ROC-AUC and Wilson rate intervals. See `reports/phase6/fairness_metrics.json`.
- Gender selection-rate gap=0.000; AgeBand selection-rate gap=0.001. Similar positive rates are **vacuous** for an all-positive-like classifier.
- Binary gender categories and restricted ages exclude many real identities; synthetic data cannot demonstrate equitable impact on real people. Small groups are suppressed; metrics are descriptive, not a fairness certification.

## Data quality / ethics / limitations
- `Returns` missingness was preserved by Phase 1–2; `Total Purchase Amount` differs from Price×Quantity for 249,952/250,000 rows; currency semantics unverified.
- The repository contains tracked synthetic names in legacy CSV/Parquet; no individual predictions or names are exported by this audit. Review reuse license before redistributing source data.
- No real-world validity, calibration, operational safety, or prospective forecast has been established. **Do not use for decisions affecting people.**

**AI-assistance:** AI tools contributed to draft code/tests/documentation; human verification and responsibility remain with student authors. See `docs/ai_usage_disclosure.md`.
