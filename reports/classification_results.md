# Phase 5 — W11: Retrospective Synthetic Churn Classification

> **Không phải dự đoán churn tương lai:** Nhãn `Churn` cố định theo khách, không có timestamp xác nhận.
> Dữ liệu mô phỏng; không dùng mô hình để quyết định về khách hàng thực.

## Bài toán, thời điểm, đơn vị phân tích
- Snapshot: các giao dịch trước **2023-01-01T00:00:00** (exclusive).
- 202,313 giao dịch lịch sử; **49,128 khách** đủ dữ liệu; loại 545 khách không có lịch sử trước cutoff.
- Positive (Churn=1): 9,835/49,128 (20.02%).
- Nhãn lấy từ thuộc tính khách trong dataset và **không có mốc xác lập**; đây là bài toán phân loại hồi cứu minh họa, không đủ bằng chứng cho prospectively valid churn forecasting.
- Features chỉ từ giao dịch trước cutoff; không dùng Returns, Name, ID, Gender, Age, Churn làm predictors.
- `recorded_amount_mean` là giá trị ghi nhận mô phỏng, **không phải doanh thu thực**; đơn vị tiền chưa rõ.

## Chia tập, CV và lựa chọn threshold
- Group holdout theo Customer ID: train 39,302; test 9,826 khách độc lập.
- StratifiedGroupKFold(3) on training customers only; threshold tối ưu bằng **OOF train F1**, không sử dụng test labels.
- Chọn mô hình dựa trên OOF train F1: **Random Forest**.
- Baselines gồm `DummyClassifier(prior)` (đa số không Churn) và **All-positive** (đoán mọi khách đều Churn), để tránh hiểu F1 cao do đoán hết lớp dương là tín hiệu thật.
- F1 all-positive trên train: 0.3347; chênh OOF train F1 so với baseline: **+0.000058** (dù dương rất nhỏ cũng không được coi là cải thiện hữu ích).
- Chênh F1 của model chọn so với all-positive trên **test**: **-0.000058**. Mô hình vẫn chỉ là bài thực hành, KHÔNG triển khai thực tế.

## Kết quả trên holdout
| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | Balanced Acc | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior baseline | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1971 | 0.5000 | 0.8029 |
| All-positive baseline | 0.5000 | 0.1971 | 1.0000 | 0.3293 | 0.5000 | 0.1971 | 0.5000 | 0.1971 |
| Logistic Regression | 0.0500 | 0.1971 | 1.0000 | 0.3293 | 0.4930 | 0.1912 | 0.5000 | 0.1971 |
| Random Forest | 0.3750 | 0.1971 | 0.9995 | 0.3293 | 0.4949 | 0.1965 | 0.4999 | 0.1973 |

## Confusion matrix trên holdout

Format `[TN, FP, FN, TP]`:
- **Prior baseline:** [7889, 0, 1937, 0]
- **All-positive baseline:** [0, 7889, 0, 1937]
- **Logistic Regression:** [0, 7889, 0, 1937]
- **Random Forest:** [3, 7886, 1, 1936]

## Khoảng tin cậy 95% (bootstrap khách hàng)
- F1: [0.3175, 0.3393]
- ROC_AUC: [0.4802, 0.5101]
- PR_AUC_average_precision: [0.1846, 0.2092]

## Feature engineering
- 19 đặc trưng hành vi: transaction_count, active_days, categories_count, payment_methods_count, quantity_mean, quantity_sum, unit_price_mean, recorded_amount_mean, calculated_line_value_mean, recency_days, history_span_days, category_share_books, category_share_clothing, category_share_electronics, category_share_home, payment_share_cash, payment_share_credit_card, payment_share_crypto, payment_share_paypal
- Mỗi khách một dòng. Các tỷ lệ danh mục/phương thức được tính trên lịch sử của chính khách hàng đó.
- Không fit imputer/scaler trên toàn dataset: các phép biến đổi nằm trong sklearn Pipeline và fit trên train/CV fold.

## Hạn chế và khuyến nghị
- Không diễn giải feature importance là quan hệ nhân quả.
- Nếu ROC-AUC gần 0.5 và PR-AUC gần prevalence, kết luận tín hiệu phân loại yếu thay vì tô hồng Accuracy.
- Cần event timestamp của Churn và dữ liệu as-of snapshots thật trước khi xây hệ thống dự đoán tương lai.
- Phân tích subgroup/fairness chi tiết được thực hiện ở Phase 6.
