# W11 / Phase 5 — Churn label & temporal validity assessment

## 1. Nhãn và khả năng xác định thời điểm

- `Churn` nằm ở cấp khách hàng, ghi lặp trên các dòng giao dịch. Kiểm tra sample CSV 250.000 dòng: **0 khách** có `Churn` khác nhau giữa các dòng cùng `Customer ID`.
- Dataset không cung cấp thời điểm một khách chuyển trạng thái `Churn=1`, thời điểm bắt đầu/đóng tài khoản, ngày đánh giá nhãn, hay khoảng horizon được dùng để định nghĩa Churn.
- Vì thế **không thể tuyên bố bài toán dự đoán churn trong 30/60/90 ngày tới**, không thể coi nhãn có nguồn gốc từ tương lai sau thời điểm feature cutoff.
- Bài toán hợp lệ trong phạm vi môn học: **retrospective classification of a static synthetic customer label**. Model chỉ minh họa quy trình phân loại W11.

## 2. Quan sát features ở một mốc cố định

- Cutoff quan sát chọn trước: **2023-01-01 00:00:00**, exclusive. Chỉ những giao dịch trước cutoff được dùng tạo đặc trưng.
- Khách không có giao dịch trước cutoff bị loại, không sinh feature từ giao dịch tương lai để lấp.
- Label lấy từ thuộc tính khách trong toàn bộ dataset (vì label static). Đây là **giới hạn temporal validity** vẫn chưa giải quyết được, dù features không nhìn về tương lai.
- File source chuẩn `data/processed/ecommerce_customer_clean.csv` từ Phase 2; legacy CSV không được sử dụng làm nguồn phân loại.

## 3. Leakage và đơn vị phân tích

| Nguy cơ | Biện pháp |
|---|---|
| Một khách xuất hiện ở cả train và test | Feature aggregation một dòng/khách, `GroupShuffleSplit` theo Customer ID; assert tập khách không giao nhau |
| Test/validation scale/impute toàn cục | `SimpleImputer`/`StandardScaler` đặt trong `Pipeline`; chỉ `.fit` trên train hoặc từng CV fold |
| Threshold tối ưu trên test | 3-fold `StratifiedGroupKFold` trong train tạo OOF probabilities, chọn threshold theo F1 từ **train** |
| Dùng tương lai để tạo feature | Lọc `Purchase Date < cutoff` **trước** khi aggregate |
| Dùng nhãn hoặc kết quả sau giao dịch làm feature | Không dùng `Churn`, `Returns`, `Returns_was_missing` hoặc các phép biến đổi từ chúng |
| Repeated customer identity | `Customer ID` chỉ dùng làm index/group split; không vào predictors |
| Tên/tính riêng tư và demographics | `Customer Name`, `Gender`, `Age` **không dùng làm predictors**; Gender/Age giữ ngoài X để đánh giá bias ở Phase 6 |
| Tiền tệ và diễn giải ảnh hưởng | `recorded_amount_mean` chỉ là feature mô tả số được ghi nhận, không được gọi revenue thực |

## 4. Các điều không thể xác minh từ dữ liệu này

- Không có ground truth churn cùng nhãn ở một thời điểm xác định; không thiết kế được evaluation theo rolling-origin hay prospective holdout hợp lệ.
- Chia theo khách và quan sát trước cutoff **không thay thế được một nhãn có timestamp rõ ràng**.
- Không có đại diện cho khách hàng thực, không dùng xác suất đầu ra làm quyết định kinh doanh hay can thiệp khách hàng.
- Hiệu năng gần chance-level, nếu xuất hiện, cần được báo cáo minh bạch thay vì ép tăng Accuracy.

## 5. Điều kiện để chuyển thành bài toán churn forecast thật

Cần định nghĩa label thời gian rõ ràng, ví dụ: *không có giao dịch trong 90 ngày sau ngày đánh giá*, nguồn giao dịch đầy đủ qua toàn bộ 90 ngày tương lai, và lịch sử trước ngày đánh giá. Khi đó tạo snapshot theo ngày, split theo thời gian và customer ID, giải quyết censoring, và đánh giá performance/fairness out-of-time. **Không áp dụng định nghĩa ví dụ này để tự gán lại nhãn dataset hiện tại.**
