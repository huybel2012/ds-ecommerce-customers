# Phase 5 methodological review (W11)

## Scope and source integrity

- Derived from Phase 2 cleaned CSV; no fallback to the tracked legacy CSV. Source must pass Phase 3 `validate_eda_data`, including exact `Returns_was_missing` consistency.
- The label `Churn` is static per customer and **lacks label creation/event timestamp**. Feature cutoff is 2023-01-01, but the model is only a **retrospective synthetic classification exercise**.
- Customer identity is used only for group splits, never as predictor. Name, Returns, Churn, Gender and Age are also excluded from predictors.
- `Total Purchase Amount` remains a synthetic recorded field of unresolved monetary semantics. Its average is a behavioral numeric predictor, **not verified revenue**.

## Acceptance tests

- Precutoff-only features, future-event mutation invariance, consistency of per-customer labels and metadata.
- Distinct train/test customers and group-disjoint CV folds, train-only threshold selection, imputer/scaler inside model Pipeline.
- Both all-negative/prior and all-positive baselines; F1, Precision, Recall, ROC-AUC, PR-AUC, balanced accuracy, Brier and confusion matrix. 95% customer bootstrap intervals for the selected model.
- Running Phase 5 must not overwrite CSV or figures 01–09; no model personally identifying artifact exported.
- Clean output absent from ZIP, so official acceptance requires running from Windows using the Phase 2-generated clean CSV.

## Important interpretation warning

High Accuracy (~80%) from predicting **no churn** reflects class prevalence; F1 (~0.33) from predicting **all churn** can be obtained without real discrimination. Model ROC-AUC around 0.5 or PR-AUC near class prevalence is no useful ranking ability. Threshold selection by F1 is for coursework only, not a cost-sensitive deployment plan. A tiny positive training OOF lift does not justify deployment.

## Phase 6 handoff

Audit held-out metrics by `Gender` and `Age` bands, based on withheld demographics only, with sample counts and uncertain estimates. No actual downstream decision-making system should use this synthetic demonstration.
