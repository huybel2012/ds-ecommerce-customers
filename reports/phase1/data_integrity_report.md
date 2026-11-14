# Phase 1 — Data Integrity & Validation

**Input:** `data/processed/ecommerce_customer_cleaned.csv` và `data/processed/ecommerce_clean.parquet`.
**Trạng thái Returns:** Đã khôi phục từ Parquet và đối chiếu từng giao dịch.

## Số liệu đã kiểm toán

| Phép đo | Giá trị |
|---|---:|
| Bản ghi | 250,000 |
| Khách hàng duy nhất | 49,673 |
| Khoảng ngày mua | 2020-01-01 00:15:00 tới 2023-09-15 12:24:08 |
| Trùng toàn dòng | 0 |
| Trùng khóa `(Customer ID, Purchase Date)` | 0 |
| Tổng `Total Purchase Amount` | 681,342,683 (đơn vị nguồn chưa xác minh) |
| Tổng `Product Price × Quantity` | 190,908,939 (đơn vị nguồn chưa xác minh) |
| Số dòng hai giá trị bằng nhau | 48 / 250,000 (0.0192%) |
| Tương quan giữa hai giá trị | -0.001719 |
| Tỷ lệ Churn theo khách hàng | 20.0149% |
| Tỷ lệ Churn theo dòng giao dịch | 19.9496% |
| Returns còn thiếu trong file đang đo | 47,596 (19.04%) |
| Tỷ lệ trả hàng trên dòng không thiếu trong file đang đo | 49.7861% |
| Tỷ lệ sau khi giả định missing = 0 | 40.3076% |

## Sáu khía cạnh chất lượng dữ liệu

1. **Completeness:** Returns thiếu 47,596 giá trị (19.04%); báo cáo tỷ lệ trả hàng chỉ trên bản ghi không thiếu.
2. **Accuracy:** `Total Purchase Amount` khác `Product Price × Quantity` ở 249,952 dòng. Không tự sửa giá trị gốc; chưa có quy tắc đối soát đáng tin cậy.
3. **Consistency:** Churn không nhất quán giữa các giao dịch của một khách: 0 mã; Age: 0; Gender: 0.
4. **Validity:** Age ngoài [18,70]: 0; Price <= 0: 0; Quantity <= 0: 0; Churn sai miền: 0; Returns sai miền: 0.
5. **Uniqueness:** Trùng toàn dòng: 0; trùng khóa sự kiện `(Customer ID, Purchase Date)`: 0. Khóa này chỉ là khóa kiểm toán, không được tác giả xác nhận là order ID.
6. **Timeliness:** Dataset mô phỏng có ngày mua tới 2023-09-15 12:24:08; không được suy rộng thành hành vi hiện tại của người tiêu dùng.

## Quy tắc diễn giải đã chốt cho Phase 1

- **`Total Purchase Amount`:** 'Giá trị được ghi nhận trong trường Total Purchase Amount', không tự động gọi là doanh thu thực tế hoặc đã được đối soát.
- **`line_item_value_calc`:** `Product Price × Quantity`, giá trị tính toán từ hai cột đầu vào, không chứng minh là số tiền thực thanh toán.
- **`Churn`:** tỷ lệ khách hàng duy nhất mang nhãn `Churn=1`.
- **`Returns`:** 1 là trả hàng quan sát được, 0 là không trả hàng quan sát được, NULL là không có dữ liệu. Tuyệt đối không gán NULL=0 khi tính tỷ lệ mô tả.
- **Đơn vị tiền:** chưa xác minh qua metadata gốc, nên không tự thêm ký hiệu `$` hoặc khẳng định USD.
- **Phạm vi:** dữ liệu tổng hợp bằng Faker, không được suy luận kết quả là tình hình thương mại điện tử thực tế.
- **License:** chưa xác minh từ metadata tác giả Kaggle; phải kiểm tra trước khi tái phân phối dataset.
- **Tên khách hàng:** chỉ là trường định danh do Faker tạo, mặc định bỏ khỏi output kiểm toán để giảm phơi lộ dữ liệu không cần thiết.

## Cổng nghiệm thu

- [x] Nguồn và tính chất mô phỏng được xác định.
- [x] Sai lệch hai chỉ tiêu tiền tệ được tính chính xác và cách diễn giải thận trọng được ghi lại.
- [x] Sáu khía cạnh chất lượng dữ liệu được rà soát.
- [x] Returns missing được khôi phục từ Parquet với đối chiếu khóa sự kiện 1:1.
- [ ] License và đơn vị tiền được xác nhận từ metadata của tác giả (chưa có bằng chứng trực tiếp).
- [ ] Phase 3 sẽ cập nhật dashboard và các mô hình để sử dụng metric được chốt ở trên.
