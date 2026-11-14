# Week 10: Regression (Phase 4)

**Input:** `data/processed/ecommerce_customer_clean.csv`, Phase 2 validated. **Target:** `Total Purchase Amount` in *unverified units*. Synthetic Kaggle dataset; do not call this measured USD/VND revenue. The target differs from Price × Quantity in **249,952 rows**; model results have limited substantive meaning.

**Design:** prediction at the transaction level. GroupShuffleSplit 80/20 by `Customer ID`, seed 42 (train 199,771 rows / 39,738 customers; test 50,229 rows / 9,935 customers). Customer overlap: **none**. Three-fold grouped CV inside train only; preprocessing is fit within each training fold. Numeric variables scaled, categorical one-hot encoded. No test-set tuning.

**Features:** Product Price, Quantity, Age, Product Category, Payment Method, Gender. Excluded: Returns (missing and post-purchase), Churn (post-outcome and customer attribute), Customer ID (split only), Purchase Date (not included for baseline comparability). In particular, `Returns` is not used: it is 19% missing and could occur **after** the purchase being predicted. This avoids unsupported imputation and temporal leakage.

## Holdout test metrics

| Metric | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline (mean) | 1,251.4015 | 1,445.0418 | -0.0000 |
| OLS | 1,250.1095 | 1,442.7104 | 0.0032 |
| Ridge | 1,250.1130 | 1,442.7135 | 0.0032 |
| Lasso | 1,250.1897 | 1,442.8167 | 0.0031 |

Best grouped-CV alpha: Ridge 1000, Lasso 10. Train OLS R²=0.00259; test OLS R²=0.00322.

**Baseline comparison:** OLS improves MAE over mean baseline by **+1.292** units; 95% **customer cluster bootstrap** CI [+0.635, +1.947]. If CI includes zero, do not claim predictive improvement beyond sampling uncertainty. Statistical significance of individual coefficients must not be confused with useful prediction.

## OLS inference (TRAIN, raw units)

| Metric | coefficient | se_cluster | ci_low | ci_high | p_cluster |
|---|---|---|---|---|---|
| const | 2,526.5493 | 14.9594 | 2,497.2285 | 2,555.8701 | 0.0000 |
| Product Price | -0.0263 | 0.0227 | -0.0707 | 0.0181 | 0.2455 |
| Quantity | 0.6821 | 2.2821 | -3.7909 | 5.1550 | 0.7650 |
| Age | 4.7414 | 0.2103 | 4.3293 | 5.1536 | 0.0000 |
| Product Category = Books | 5.7347 | 8.3603 | -10.6518 | 22.1212 | 0.4928 |
| Product Category = Electronics | -3.0100 | 9.3359 | -21.3085 | 15.2885 | 0.7471 |
| Product Category = Home | -17.8560 | 9.3059 | -36.0958 | 0.3838 | 0.0550 |
| Payment Method = Cash | -3.0510 | 8.7886 | -20.2768 | 14.1748 | 0.7285 |
| Payment Method = Crypto | -7.3060 | 11.5354 | -29.9156 | 15.3036 | 0.5265 |
| Payment Method = PayPal | -1.5884 | 7.8111 | -16.8983 | 13.7215 | 0.8389 |
| Gender = Male | -1.9707 | 6.4656 | -14.6434 | 10.7020 | 0.7605 |

Standard errors: **CR1 cluster sandwich by Customer ID** (not ordinary iid OLS or HC3). This accounts for within-customer repeated transactions under the working assumption that different customers are independent. Data are observational and synthetic, not a causal experiment.

### Coefficient interpretations

- `Product Price`: -0.026 recorded units (95% customer-cluster CI [-0.071; +0.018]); CI includes zero. Conditional association, **not causal**.
- `Quantity`: +0.682 recorded units (95% customer-cluster CI [-3.791; +5.155]); CI includes zero. Conditional association, **not causal**.
- `Age`: +4.741 recorded units (95% customer-cluster CI [+4.329; +5.154]); CI excludes zero. Conditional association, **not causal**.
- `Product Category = Books`: +5.735 recorded units (95% customer-cluster CI [-10.652; +22.121]); CI includes zero. Conditional association, **not causal**.
- `Product Category = Electronics`: -3.010 recorded units (95% customer-cluster CI [-21.309; +15.289]); CI includes zero. Conditional association, **not causal**.
- `Product Category = Home`: -17.856 recorded units (95% customer-cluster CI [-36.096; +0.384]); CI includes zero. Conditional association, **not causal**.
- `Payment Method = Cash`: -3.051 recorded units (95% customer-cluster CI [-20.277; +14.175]); CI includes zero. Conditional association, **not causal**.
- `Payment Method = Crypto`: -7.306 recorded units (95% customer-cluster CI [-29.916; +15.304]); CI includes zero. Conditional association, **not causal**.
- `Payment Method = PayPal`: -1.588 recorded units (95% customer-cluster CI [-16.898; +13.722]); CI includes zero. Conditional association, **not causal**.
- `Gender = Male`: -1.971 recorded units (95% customer-cluster CI [-14.643; +10.702]); CI includes zero. Conditional association, **not causal**.

## Diagnostics and limitations

- Cook's D > 4/n: 1,499 observations (flag for review, not automatic deletion); max VIF 1.40.
- Residual, QQ, Cook's influence diagnostics in `reports/figures/06_residual_diagnostics.png`; 95% customer-cluster coefficient CIs in `07_coefficients_ci.png`; holdout actual vs prediction in `08_actual_vs_predicted.png`.
- **Durbin–Watson omitted:** sorting unrelated customers' transactions by date does not create a meaningful single time series, so the old temporal-DW interpretation was invalid.
- Unverified target semantics and synthetic generation limit external validity. A low R² is itself an empirical finding: do not optimize on test data or promise an unrealistically high score.
- Cluster-robust inference does not repair confounding, selection effects, or an incorrectly defined prediction target.
