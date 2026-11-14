# Week 9: Statistical inference (Phase 4)

**Input:** `data/processed/ecommerce_customer_clean.csv` (Phase 2 validated); **250,000 transactions / 49,673 customers**. Synthetic Kaggle data, monetary unit unverified; `Total Purchase Amount` cannot be reconciled with `Price × Quantity`. This is exploratory retrospective analysis: the existing hypotheses were inspected on this dataset, so do **not** present as preregistered confirmatory research. No causal or market-wide claims.

**Multiplicity:** Two-sided α=0.05; family of 3 primary hypotheses controlled by **Holm**. The H3 post-hoc family of six category pairs separately uses Holm-adjusted p-values and *Bonferroni simultaneous* confidence intervals. Avoid cherry-picking the widest observed gap.

## H1: Gender and customer churn
- **H0:** p(Male churn) = p(Female churn); **H1:** different. Unit: customer, deduplicated by `Customer ID`.
- Male: 20.295% (n=24,715), Female: 19.737% (n=24,958).
- Difference (male − female): **+0.558 percentage points**, 95% unadjusted CI [-0.146, +1.262] pp; Cohen's h=0.0140.
- Two-proportion z=1.555; p=0.12; **Holm p=0.24**. Chưa bác bỏ H0 at α=.05. Statistical non-significance does not prove equality.

## H2: Per-customer average transaction amount by churn status
- **H0:** average of customer-level mean recorded amounts is equal between Churn=1 and 0; **H1:** different. Unit: customer, not independent transaction rows.
- Churn: 2,727.89 (n=9,942), retained: 2,723.83 (n=39,731).
- Difference: **+4.06** (unverified units), Welch 95% CI [-12.17, +20.29], Cohen's d=0.0055.
- Welch t=0.490, df=15263, p=0.6238; **Holm p=0.6238**. Chưa bác bỏ H0 at α=.05. Association only.

## H3: Mean recorded amount across four categories
- **H0:** 4 category means are equal; **H1:** at least one differs.
- Unit: transaction; repeated transactions per customer are explicitly handled with **customer-clustered sandwich SE** and a chi-square Wald omnibus test (3 df, asymptotic approximation with 49,673 independent customer clusters).
- Category means: Books: 2,735.74 (n=74,912), Clothing: 2,725.21 (n=75,052), Electronics: 2,721.92 (n=50,185), Home: 2,713.51 (n=49,851).
- Wald χ²=7.570, p=0.05579; **Holm p=0.1674**. Chưa bác bỏ H0 at α=.05.
- Descriptive η²=0.0000301. Ordinary row-level ANOVA p=0.05696 is **diagnostic only**, because rows are not independent customers.

### Exploratory pairwise comparisons (6 pairs)

| Pair | Mean difference | 95% simultaneous CI (Bonferroni) | Holm p |
|---|---:|---|---:|
| Books vs Clothing | +10.53 | [-9.20, +30.25] | 0.6362 |
| Books vs Electronics | +13.82 | [-8.12, +35.76] | 0.483 |
| Books vs Home | +22.23 | [+0.28, +44.18] | 0.04527 |
| Clothing vs Electronics | +3.29 | [-18.81, +25.40] | 0.7152 |
| Clothing vs Home | +11.70 | [-10.30, +33.70] | 0.6362 |
| Electronics vs Home | +8.41 | [-15.70, +32.52] | 0.7152 |

**Notes:** The six comparisons were enumerated before selecting a largest observed gap. These adjusted intervals are valid for the pre-defined six-pair family under the large-cluster approximation, not proof of an economically meaningful effect. The category exposure is observational; no causal interpretation.

## Methodology and limits
- The dataset is synthetic; the statistical p-values describe this generated sample under working sampling assumptions, not the real retail population.
- Neither gender nor churn are assigned treatments; these comparisons cannot establish causal effects.
- `Returns` missingness (47,596 on canonical input) does not affect these tests, which do not use Returns.
- The transaction amount is a *recorded field with unresolved semantics* rather than confirmed revenue; do not label results as USD or VND.
- `reports/figures/09_hypothesis_tests.png` uses a zero-based churn bar chart and dot/CI plots instead of truncated bars for tiny differences.
