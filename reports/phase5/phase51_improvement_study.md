# Phase 5.1 — Classification Improvement Study

> **Nghiên cứu thăm dò trên TRAIN ONLY.** Không ghi đè Phase 5; chưa đánh giá lại holdout đã được xem.
> Nhãn Churn synthetic không có event timestamp. Không dự đoán rời bỏ tương lai.

## Phạm vi và giao thức
- 250,000 transactions; 49,128 customers eligible; train=39,302; original holdout not scored=9,826.
- Cutoff exclusive: `2023-01-01T00:00:00`; giữ nguyên split theo khách của Phase 5.
- 19 original features vs 28 after 9 temporal features.
- So sánh Logistic Regression và HistGradientBoosting trên 2 feature sets bằng repeated, group-disjoint CV.
- ROC-AUC, average precision (PR-AUC), top-decile lift. Không tối ưu F1 threshold hay dùng test labels để chọn mô hình.
- Negative control: xáo trộn nhãn trong tập train, duy trì phân bố lớp.

## Kết quả repeated grouped CV (chỉ train)

| Feature set / model | ROC-AUC | PR-AUC (AP) | AP - prevalence | Lift@10% |
|---|---:|---:|---:|---:|
| Phase 5 (19 features) | Logistic Regression | 0.4993 | 0.2016 | +0.0006 | 1.002 |
| Phase 5 (19 features) | HistGradientBoosting | 0.4935 | 0.1985 | -0.0025 | 0.982 |
| Phase 5.1 (28 features) | Logistic Regression | 0.4960 | 0.1996 | -0.0014 | 0.970 |
| Phase 5.1 (28 features) | HistGradientBoosting | 0.4906 | 0.1967 | -0.0042 | 0.942 |

**Ứng viên có ROC-AUC CV lớn nhất:** `Phase 5 (19 features) | Logistic Regression` (chọn trên train, có selection bias).
**Negative control mean ROC-AUC:** 0.5026.
**Train-screen:** FAIL.
**Quyết định:** No robust train-only evidence of improved churn discrimination; retain Phase 5 report.

## Pre-registered screening criteria

- FAIL: `ROC_AUC_at_least_0_55`
- FAIL: `AP_gain_at_least_0_03`
- FAIL: `top_decile_lift_at_least_1_20`
- FAIL: `AUC_gain_vs_permutations_at_least_0_03`
- FAIL: `all_repeat_AUC_at_least_0_52`

## Tín hiệu univariate mạnh nhất trên train (thăm dò)
| Feature | |AUC-0.5| | Mutual information |
|---|---:|---:|
| `quantity_sum` | 0.0060 | 0.0000 |
| `active_days` | 0.0058 | 0.0000 |
| `std_gap_days` | 0.0057 | 0.0000 |
| `transaction_count` | 0.0055 | 0.0000 |
| `unit_price_mean` | 0.0053 | 0.0000 |
| `calculated_line_value_mean` | 0.0050 | 0.0000 |
| `categories_count` | 0.0041 | 0.0002 |
| `payment_share_cash` | 0.0040 | 0.0000 |
| `payment_share_crypto` | 0.0032 | 0.0000 |
| `category_switch_rate` | 0.0032 | 0.0000 |

## Giới hạn và kết luận

- Static synthetic Churn has no event/effective timestamp; never claim future prediction.
- Previous Phase 5 holdout was already inspected; it is not a fresh confirmatory test.
- Screening four model/feature candidates produces selection optimism even within CV.
- Time-derived features may be sparse because transaction history per customer is short.
- Recorded amount units and interpretation are unresolved; not verified revenue.
- Negative-control permutations assess spurious fit but cannot prove label-generation mechanism.
- No Returns, customer names, identifiers, age, gender, or post-cutoff data are predictors.

**Phase 5 original remains authoritative unless a new, independently validated evaluation justifies a change.**
