# Phase 6 — Ethics, Fairness & Responsible AI (W12)

## Scope & method

- Synthetic Faker Kaggle dataset. Not real-world customer fairness evidence.
- Phase 5 **frozen Random Forest** and its training-only-selected threshold; exact customer holdout reproduced.
- Sensitive `Gender`/`Age` used ONLY for post-hoc audit, not model training; one customer = one observation.
- Raw/reconstructed source is validated and hashed. No customer identifiers/names/probabilities are exported.
- Frozen classification decisions must match Phase 5 exactly; ranking AUC/AP allowed a documented absolute tolerance of 0.005 for library differences, with every discrepancy logged (NOT claimed as exact).
- Gender: Male/Female only; age bins: 18-25 / 26-35 / 36-45 / 46-55 / 56-65 / 66-70.
- Rates: TPR=TP/(TP+FN); FPR=FP/(FP+TN); PPV=TP/(TP+FP); selection_rate=(TP+FP)/n.
- Wilson 95% intervals for binomial rates. Suppress cells with insufficient group/class support.

## Overall holdout results

- N=9,826; fixed threshold=0.375; ROC-AUC=0.4949, PR-AUC=0.1965, balanced accuracy=0.4999.
- TN/FP/FN/TP=[3, 7886, 1, 1936]. Predictive validity is extremely limited.
- **Phase 5 replay:** ranking metric deltas: {'ROC_AUC': 0.0, 'PR_AUC_average_precision': 0.0, 'F1': 0.0} (current minus committed Phase 5).

## Subgroup performance

### Gender (dataset categories)

| Group | N | Churn=1 | Predicted-positive rate | TPR | FPR | PPV | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Female | 4901 | 950 | 1.000 | 1.000 | 1.000 | 0.194 | 0.483 |
| Male | 4925 | 987 | 0.999 | 0.999 | 0.999 | 0.200 | 0.506 |

**Descriptive gaps:** selection_rate_gap=0.000, tpr_gap=0.001, fpr_gap=0.000, ppv_gap=0.006, equalized_odds_gap=0.001

### Age bins

| Group | N | Churn=1 | Predicted-positive rate | TPR | FPR | PPV | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 18-25 | 1523 | 318 | 0.999 | 1.000 | 0.999 | 0.209 | 0.508 |
| 26-35 | 1882 | 337 | 1.000 | 1.000 | 1.000 | 0.179 | 0.494 |
| 36-45 | 1832 | 347 | 0.999 | 1.000 | 0.999 | 0.190 | 0.506 |
| 46-55 | 1847 | 353 | 0.999 | 1.000 | 0.999 | 0.191 | 0.478 |
| 56-65 | 1823 | 379 | 0.999 | 0.997 | 1.000 | 0.207 | 0.506 |
| 66-70 | 919 | 203 | 1.000 | 1.000 | 1.000 | 0.221 | 0.461 |

**Descriptive gaps:** selection_rate_gap=0.001, tpr_gap=0.003, fpr_gap=0.001, ppv_gap=0.042, equalized_odds_gap=0.003

### Intersectional Gender × Age

| Group | N | Churn=1 | Predicted-positive rate | TPR | FPR | PPV | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Female/18-25 | 726 | 144 | 0.999 | 1.000 | 0.998 | 0.199 | 0.522 |
| Female/26-35 | 956 | 175 | 1.000 | 1.000 | 1.000 | 0.183 | 0.497 |
| Female/36-45 | 921 | 176 | 1.000 | 1.000 | 1.000 | 0.191 | 0.499 |
| Female/46-55 | 924 | 172 | 1.000 | 1.000 | 1.000 | 0.186 | 0.472 |
| Female/56-65 | 915 | 187 | 1.000 | 1.000 | 1.000 | 0.204 | 0.470 |
| Female/66-70 | 459 | 96 | 1.000 | 1.000 | 1.000 | 0.209 | 0.431 |
| Male/18-25 | 797 | 174 | 1.000 | 1.000 | 1.000 | 0.218 | 0.499 |
| Male/26-35 | 926 | 162 | 1.000 | 1.000 | 1.000 | 0.175 | 0.491 |
| Male/36-45 | 911 | 171 | 0.999 | 1.000 | 0.999 | 0.188 | 0.513 |
| Male/46-55 | 923 | 181 | 0.999 | 1.000 | 0.999 | 0.196 | 0.484 |
| Male/56-65 | 908 | 192 | 0.999 | 0.995 | 1.000 | 0.211 | 0.542 |
| Male/66-70 | 460 | 107 | 1.000 | 1.000 | 1.000 | 0.233 | 0.487 |

**Descriptive gaps:** selection_rate_gap=0.001, tpr_gap=0.005, fpr_gap=0.002, ppv_gap=0.058, equalized_odds_gap=0.005

## Interpretation & limitations

- **No fairness certification.** Similar selection rates can be trivial because nearly everyone is predicted positive, not evidence of equitable real-world impact.
- Differences in TPR/FPR/PPV are sample descriptions only; CI widths and group prevalence need to be read alongside point estimates.
- The label has no effective date, ROC-AUC is near chance, and training data is synthetic. Deployment is prohibited by this project protocol.
- Small-group suppression prevents sparse-cell overinterpretation and avoids releasing tiny subgroup confusion matrices.
- Gender categories exclude non-binary and unknown identities; age limited to 18–70. External validity not established.
- Tracked legacy CSV/Parquet include synthetic `Customer Name`; never claim full de-identification or release new raw extracts.

## Governance & licensing

- Source: https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis
- Dataset reuse license: **NOT independently verified from the primary Kaggle license field**. Do not assert CC0/MIT or unrestricted redistribution.
- Kaggle downloader should check current license, attribution and derivative redistribution terms before public distribution.
- Do not use synthetic results to target, penalize, price-discriminate, or exclude actual people.
- Ethical review is ongoing if legal permission, source semantics or stakeholder approval remains unverified.

## Data operations & scalability

- The data has 250k rows (~50k customers): Pandas/Parquet and a local Python 3.11 pipeline are sufficient for the demonstrated study.
- No need to add Spark solely for W12; at larger scale prefer chunked/partitioned Parquet, key validation, memory profiling, orchestration and distributed joins only when measured memory/latency requirements demand them.

## AI usage & human accountability

- See `docs/ai_usage_disclosure.md`: AI-assisted code/report drafting and tests must be reviewed by student authors; no automated assistant guarantees correctness, legality, or ethics.

## Decision

**DO NOT DEPLOY.** Phase 6 measures limitations and subgroup discrepancies; it does not repair the weak predictive signal in Phase 5–5.1.
