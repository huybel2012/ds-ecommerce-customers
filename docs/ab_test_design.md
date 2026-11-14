# Phase 4 · W9: Proposed A/B Test (design only)

**Status: protocol, NOT an experiment performed on the Kaggle CSV.** The synthetic purchase-only dataset has no random assignment, no user exposure log, no visitors who did not purchase, and no verified temporal definition of Churn. No treatment effect, conversion rate or causal A/B result can be estimated from it.

## Question, units and intervention

**Decision question:** Does a simpler post-purchase reorder interface increase a *customer's* probability of placing a new order within 30 days after a qualifying completed purchase?

- **Eligible population:** real consented customers completing an eligible purchase, with consent and instrumentation consistent with the shop's policy. This requires a **future real system**, not this dataset.
- **Randomization unit:** `Customer ID`, not individual purchase or page view. Hash assignment once, persist it and use 1:1 allocation throughout the 30-day observation. Prevent customer contamination across groups and do not re-randomize repeated visits.
- **Control A:** current reorder interface. **Treatment B:** simplified one-click reorder entry point, without changing price, eligibility or fulfillment.
- **H0:** 30-day repeat-purchase probability is the same. **H1:** different (two-sided). Prefer reporting absolute percentage-point difference and 95% CI, not only relative lift.
- **Outcome window:** 30 days after the first eligible purchase that activated the assigned experience. Follow all assigned customers for that full window. Do not count multiple reorders from one customer as independent Bernoulli outcomes.
- **Primary metric:** share of all randomized eligible customers with at least one *completed new purchase* within 30 days; denominator is randomized eligible customers, analyzed by intention to treat. Define deduplication of events and completed status before launch.
- **Guardrails:** refund/complaint counts with independently instrumented completeness, opt-outs, app error rate, and latency; monitor data loss for both arms. The Kaggle `Returns` field has ~19% missing and MUST NOT be used as an assumed complete guardrail baseline.

## Sampling and power (illustrative assumptions, not measured baselines)

Two-sided alpha=0.05, target power=80%, allocation 1:1, **assumed** control p0=20% and minimum worthwhile effect +2 percentage points (p1=22%). Normal two-proportion approximation:

\[
 n_{\mathrm{per\ arm}} \approx \frac{\left[z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{1-\beta}\sqrt{p_0(1-p_0)+p_1(1-p_1)}\right]^2}{(p_1-p_0)^2}.
\]

Using `src.ab_design.required_per_arm(0.20, 0.22)` gives **6,510 customers per arm**, ~13,020 total. This ignores attrition; for illustrative 10% unusable outcome tracking, recruit **ceil(6510 / 0.90) = 7,234 per arm**, ~14,468 total. Validate actual baseline rate, MDE, eligibility and observed event completeness with a separate pilot. Do NOT substitute the synthetic dataset's ~20% *Churn* for a measured *repeat purchase* baseline.

## Operation and analysis

1. Freeze eligibility, tracking, business metrics, statistical test, excluded accounts and alpha before randomization.
2. Run A/A instrumentation test. Verify experiment assignment, deduplicated event timestamps, bot exclusion, and Sample Ratio Mismatch (e.g. chi-square p < .001 as alert threshold); investigate tracking errors before inference.
3. Log immutable assignment (`Customer ID`, arm, assigned timestamp); count missing outcomes and attrition by arm; keep a separate audit table.
4. Run until the precommitted sample size is reached AND each customer completes the 30-day observation. No unplanned repeated significance peeks or early stopping.
5. Estimate difference in customer-level 30-day proportions with CI and two-sided test. Report both arm sample sizes, dropout, guardrails and any deviations. Under missing outcome data, perform sensitivity analysis rather than treating missing as zero.
6. Rollout only if the effect is practically important, the CI is acceptable, and predefined guardrails remain healthy. Document ethics, consent and fairness before deployment.

**Scope:** a prospective proposal only; not evidence of historical treatment effects, since Kaggle observations lack assignment and exposure data.
