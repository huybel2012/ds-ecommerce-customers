# Phase 1 — Audit Snapshot (2026-11-14)

**Repo baseline:** `b9ba18c` (`Dung` đồng bộ `main` tại thời điểm kiểm toán).
**Dữ liệu được kiểm toán:** `data/processed/ecommerce_customer_cleaned.csv`, `data/processed/ecommerce_clean.parquet` thuộc ZIP do nhóm cung cấp.

## 1. Các bằng chứng đã xác minh

| Phát hiện | Bằng chứng | Ý nghĩa |
|---|---:|---|
| Dòng giao dịch trong CSV | 250.000 | Một khách có nhiều giao dịch |
| Khách hàng duy nhất | 49.673 | Phải tính churn cấp khách |
| Cột CSV sau cleaning | 12 | Không có indicator cho Returns |
| Toàn bộ NULL trong CSV | 0 | CSV cũ đã mất thông tin thiếu của Returns |
| **NULL của `Returns` trong Parquet** | **47.596** | Đọc trực tiếp `null_count` từ Parquet footer / ColumnMetaData / Statistics; chiếm **19,0384%** |
| Giá trị Returns=1 trong CSV | 100.769 | Đây là các dòng quan sát trả hàng; dữ liệu cũ fill NULL=0 |
| Số Returns quan sát gốc theo nguồn Parquet | 202.404 | 250.000 − 47.596 |
| **Tỷ lệ trả hàng trên dòng quan sát gốc** | **49,79%** | 100.769 / 202.404, cần đối chiếu lại nội dung từng dòng khi chạy recovery |
| Tỷ lệ trả hàng sau khi fill NULL=0 | **40,31%** | 100.769 / 250.000, không được mô tả là tỉ lệ quan sát gốc |
| Tổng `Total Purchase Amount` | 681.342.683 | Theo tên trường dataset; đơn vị tiền chưa được xác minh |
| Tổng `Product Price × Quantity` | 190.908.939 | Giá trị tính từ hai cột đơn giá và số lượng |
| Dòng hai đại lượng bằng nhau | 48 / 250.000 | Khác nhau ở 249.952 dòng |
| Pearson của hai đại lượng | −0,001719 | Không có quan hệ tuyến tính đáng kể trong mẫu |
| Churn theo unique customer | 20,0149% | Khác 19,9496% nếu tính theo dòng giao dịch |
| Trùng toàn dòng | 0 | Kiểm tra CSV |
| Trùng `(Customer ID, Purchase Date)` | 0 | Khóa sự kiện 1:1 trong CSV hiện tại |
| ID có Churn/Age/Gender không nhất quán | 0 / 0 / 0 | Nhất quán trong CSV hiện tại |
| Tuổi ngoài [18,70], Price <=0, Quantity <=0 | 0 / 0 / 0 | Không phát hiện vi phạm miền dữ liệu này |
| Ngày mua | 2020-01-01 → 2023-09-15 | Mô phỏng dữ liệu lịch sử; tháng cuối chưa đầy đủ |

**Nguồn của số liệu:** phép tính Pandas trên CSV của repo; metadata Parquet (footer, Thrift compact) từ bản Parquet trong ZIP. Việc đọc `null_count` **không tương đương đọc/đối chiếu toàn bộ nội dung cột**. Bước đối chiếu từng giao dịch được cài đặt trong `src/phase1_data_integrity.py` và cần chạy với môi trường có `pyarrow`.

## 2. Đánh giá theo sáu chiều chất lượng

| Dimension | Kết quả | Nhận định |
|---|---|---|
| Completeness | Returns: 19,04% gốc thiếu | **WARN:** không gán thiếu = 0 cho phân tích mô tả |
| Accuracy | 99,9808% Total khác Price×Quantity | **BLOCK** đối với các tuyên bố doanh thu đã đối soát; không có bằng chứng nguyên nhân |
| Consistency | Customer Age/Age được ghi trùng trong cleaning log; ID thống nhất về Churn/Age/Gender | PASS trên dữ liệu hiện có; chạy lại check trong script recovery |
| Validity | Giá, lượng, tuổi, Churn/Returns không ngoài miền dữ liệu đã kiểm tra | PASS trên các rules đã khai báo, không đồng nghĩa xác minh thực tế |
| Uniqueness | Không trùng dòng, không trùng key tạm `ID+Timestamp` | PASS cho CSV hiện tại; không tự khẳng định có order ID thật |
| Timeliness | 2020–2023, dữ liệu synthetic | **LIMITATION:** không suy ra xu hướng thương mại điện tử 2026 |

## 3. Nguyên tắc ra quyết định

- Giữ nguyên **cả hai** đại lượng tiền, gắn tên rõ ràng `reported_amount` và `line_item_value_calc` khi trình bày; không ghi đè dữ liệu.
- Vì không có metadata xác nhận currency hoặc các khoản điều chỉnh (thuế, giảm giá, shipping), **không tuyên bố số tiền là USD hoặc tổng doanh thu thực nhận**.
- Khôi phục Returns từ Parquet qua khóa `(Customer ID, Purchase Date)` có kiểm tra duy nhất, kiểm tra các trường khác trùng và kiểm tra các Returns quan sát được chưa từng bị đổi. Nếu đối chiếu không đạt, **dừng xử lý**, không tạo file mới.
- Không khẳng định MCAR chỉ vì missing rate tương đương theo Payment Method và Product Category.
- Giữ ID khách hàng cho phân tích cấp khách; giảm phát tán `Customer Name` sinh giả lập trong output mới.

## 4. Vấn đề chưa thể nghiệm thu ngay từ ZIP

1. **License của Kaggle:** Chưa xác minh trực tiếp từ metadata của chính dataset. Cần xác minh trên Kaggle trước khi phát hành bản raw/processed rộng rãi.
2. **Currency và ngữ nghĩa Total:** Mô tả nguồn chưa đủ giải thích sự chênh lệch với Price×Quantity. Không tự tạo đáp án không có bằng chứng.
3. **Recovery Returns cuối cùng:** Đã viết và kiểm thử logic bằng unit tests, nhưng môi trường kiểm toán này thiếu `pyarrow`, nên chưa thực thi toàn bộ phép join trên Parquet thật. Máy Windows của nhóm **đã cài pyarrow**; chạy theo `docs/phase1_runbook.md` để nghiệm thu.
4. **Dashboard:** Hiện vẫn dùng CSV cũ và chưa được cập nhật. Theo roadmap, đây là công việc của Phase 3, sau khi pipeline nguồn được khôi phục ở Phase 2.

## 5. Nguồn bên ngoài

- Kaggle dataset: https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis
- Bản mô tả dataset/Faker: https://github.com/AriesRmd/E-commerce-Customer-Data-For-Behavior-Analysis
- Không sử dụng giấy phép một notebook Kaggle hay repository thứ ba thay cho license của dataset gốc.
