# Phase 7 — Repository Delivery Audit (W1–W12)

**Scope:** Documentation, reproducibility, defense and release checks; **no merge or PR**.
**Technical readiness:** PASS (automated checks only).
**Public redistribution:** NOT CLEARED. Do not infer data license, consent or anonymization.
**Merge:** deferred by request.

## Automated checks

| Check | Result |
|---|---|
| `required_documents_present` | PASS |
| `frozen_evidence_present` | PASS |
| `all_14_figures_present` | PASS |
| `integrity_data_counts` | PASS |
| `w9_holm_family_result` | PASS |
| `w10_regression_result` | PASS |
| `w11_frozen_holdout` | PASS |
| `phase51_no_false_upgrade` | PASS |
| `w12_frozen_fairness_holdout` | PASS |
| `ai_disclosure_in_readme` | PASS |
| `no_main_merge_in_delivery_instructions` | PASS |

## Frozen result cross-check

```json
{
  "transactions": 250000,
  "customers": 49673,
  "returns_missing": 47596,
  "amount_mismatches": 249952,
  "w9_holm_p": {
    "H1": 0.240046,
    "H2": 0.6238,
    "H3": 0.167378
  },
  "w10_ols_test_R2": 0.003222147050145896,
  "w10_ols_mae_gain": 1.2919325441773708,
  "w11_model": "Random Forest",
  "w11_threshold": 0.375,
  "w11_roc_auc": 0.4948531813344853,
  "w11_confusion_matrix_tn_fp_fn_tp": [
    3,
    7886,
    1,
    1936
  ],
  "w12_holdout_customers": 9826
}
```

## Tracked legacy datasets (manual disposition)

- `data/processed/ecommerce_customer_cleaned.csv`
- `data/processed/ecommerce_clean.parquet`

## Gates needing human approval

- [ ] Dataset Kaggle license/redistribution rights: verify from primary source, record evidence.
- [ ] Legacy tracked CSV/Parquet contain synthetic Customer Name/ID: approve publication or history remediation.
- [ ] No measured k-anonymity claim; no real-data consent/privacy clearance.
- [ ] Team identity, contribution evidence, AI-use statement and one-page model card need human sign-off.
- [ ] Final Windows run on canonical clean data and CI run at Phase 7 commit require confirmation.

This audit is evidence management, not an independent rerun of the 250k-row pipeline.
