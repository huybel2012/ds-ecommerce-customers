# Data Dictionary — Phase 1: Data Integrity & Validation

**Dataset**: [E-commerce Customer Data For Behavior Analysis, Shriyash Jagtap, Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)
**Chọn file theo README nhóm:** `ecommerce_customer_data_custom_ratios.csv`.
**Nguồn tham chiếu về cách tạo:** [mô tả dataset tương ứng trên GitHub](https://github.com/AriesRmd/E-commerce-Customer-Data-For-Behavior-Analysis) (synthetic, tạo bằng Faker).
**Cảnh báo nguồn:** Không dùng dữ liệu để suy rộng về khách hàng thực hay thị trường thực. Cần kiểm tra trực tiếp giấy phép sử dụng từ Kaggle trước khi xuất bản/tái phân phối bộ dữ liệu. Không suy đoán license từ các dataset khác.

## Đơn vị quan sát và các khóa

- **Mỗi dòng:** một bản ghi mua hàng được tạo giả lập trong dataset (KHÔNG phải một khách hàng duy nhất).
- **`Customer ID`:** định danh khách hàng; một khách có nhiều giao dịch.
- **Khóa kiểm toán:** `(Customer ID, Purchase Date)` hiện không trùng trong bản CSV 250.000 dòng; chỉ dùng làm khóa đối chiếu có xác minh trong Phase 1. Dataset **không cung cấp order ID chuẩn tắc**, nên không khẳng định khóa này luôn duy nhất với dữ liệu khác.
- **Mô hình churn:** `Churn` không thay đổi giữa các dòng của cùng một Customer ID trong mẫu đã kiểm tra. Tuy nhiên, *thời điểm định nghĩa nhãn Churn chưa rõ*, cần xác minh trước W11.

## Các cột dữ liệu gốc

| Column | Ý nghĩa theo nguồn | Phạm vi/giá trị trong bản đang dùng | Lưu ý chất lượng |
|---|---|---|---|
| `Customer ID` | Mã khách hàng | 1–50.000; 49.673 unique | Không phải mã đơn hàng |
| `Purchase Date` | Thời điểm mua | 2020-01-01 tới 2023-09-15 | Có thời gian; 2023-09 là tháng chưa đủ |
| `Product Category` | Danh mục sản phẩm | Books, Clothing, Electronics, Home | Bốn nhóm |
| `Product Price` | Giá đơn vị sản phẩm | 10–500 | **Đơn vị tiền chưa xác minh** |
| `Quantity` | Số sản phẩm | 1–5 | Số nguyên dương |
| `Total Purchase Amount` | Trường tổng tiền theo mô tả của tác giả | 100–5.350 | **Không đối soát với Price × Quantity ở 249.952 dòng**, không kết luận đây là doanh thu đã thu thật |
| `Payment Method` | Cách thanh toán | Credit Card, PayPal, Cash, Crypto | Chưa có phí/phụ thu/tỷ giá |
| `Returns` | Trả hàng | 0, 1, NULL trong file Parquet | **47.596 NULL trong Parquet**; CSV cũ đã điền NULL=0 |
| `Customer Name` | Tên được Faker tạo | Chuỗi tên giả lập | Loại khỏi output mới vì không cần cho phân tích |
| `Customer Age` | Tuổi | 18–70 | Trùng với `Age` theo cleaning log, loại ở cleaned |
| `Age` | Tuổi khách hàng | 18–70 | Dùng một cột tuổi duy nhất |
| `Gender` | Nhóm giới tính do dataset gán | Male, Female | Chỉ có hai nhóm trong dữ liệu, không đại diện đầy đủ |
| `Churn` | Nhãn khách hàng rời bỏ | 0, 1 | Thuộc tính cấp khách hàng; định nghĩa thời gian chưa đủ |

## Các cột do pipeline tạo

| Column | Công thức/nguồn | Ý nghĩa và quy tắc sử dụng |
|---|---|---|
| `Returns_was_missing` | `Returns.isna().astype('int8')` trước imputation | 1 khi dữ liệu Returns gốc thiếu; không dùng để suy ra khách trả hàng hay không |
| `line_item_value_calc` | `Product Price * Quantity` | Giá trị tính được theo dòng, KHÔNG tự động là tiền thực thanh toán |
| `revenue_calc` | Cột phụ trong Parquet cũ: `Price × Quantity` | Tên cột lịch sử có thể gây hiểu lầm; trong output Phase 1 dùng `line_item_value_calc` thay thế |

## Định nghĩa chỉ số (metric contract)

1. `reported_amount_total = sum(Total Purchase Amount)`: tổng **giá trị được ghi nhận trong trường dữ liệu**, không tự động đặt tên "doanh thu". Đơn vị tiền chưa được tác giả xác minh trong metadata có thể truy cập.
2. `line_item_value_total = sum(Product Price * Quantity)`: tổng số tiền theo công thức giá × lượng, **chỉ là phép tính**. Chênh lệch lớn với `reported_amount_total` không đủ căn cứ để kết luận nguyên nhân là giảm giá, thuế, phí hay lỗi dữ liệu.
3. `n_transactions = len(df)`: số **dòng giao dịch**, chưa có order ID độc lập.
4. `n_customers = df['Customer ID'].nunique()`: số khách hàng duy nhất.
5. `churn_rate = mean(Churn sau drop_duplicates('Customer ID'))`: tỷ lệ churn **ở cấp khách hàng**; không dùng trung bình trên dòng giao dịch khi báo cáo churn cấp khách.
6. `return_rate_observed = mean(Returns.dropna())`: tỷ lệ trả hàng trên **những bản ghi quan sát được**; phải kèm `n_observed`, `n_missing` và lưu ý selection bias.
7. `return_rate_missing_zero` chỉ là **phân tích độ nhạy khi giả định NULL=0**, không được đặt tên "tỷ lệ trả hàng thực tế".
8. Trục thời gian: tổng tháng 2023-09 là tháng chưa đầy đủ, so sánh trực tiếp với tháng đầy đủ dễ gây lệch.

## Nguyên tắc dữ liệu của nhóm

- Giữ nguyên `Total Purchase Amount` và `Product Price`; không ghi đè để hai cột "khớp nhau".
- Giữ thiếu `Returns` ở dạng NULL trong nguồn phục hồi; không impute cho mục đích thống kê mô tả.
- Chỉ khi xây mô hình, nếu bắt buộc phải impute, cần làm trong pipeline fit train-only và giữ missing indicator.
- Giảm trường định danh trong tệp công khai; `Customer Name` không cần cho biểu đồ hoặc mô hình hiện tại.
- Các quan hệ thống kê trên dữ liệu mô phỏng **không có ý nghĩa khuyến nghị kinh doanh ngoài đời thực** trừ khi kiểm chứng trên dữ liệu thực.

## Vấn đề chưa được nguồn xác minh

- **License:** Chưa lấy được metadata license trực tiếp từ trang dataset trên Kaggle; không được tự ghi CC0/MIT/Apache cho dataset này.
- **Currency:** Không đủ bằng chứng về loại tiền/tỷ giá; tránh hiển thị `$` như sự thật đã xác minh.
- **Total Purchase Amount semantics:** Nguồn định nghĩa là amount per transaction nhưng không giải thích tại sao khác Price × Quantity gần 100% dòng.
- **Returns missingness mechanism:** Tỷ lệ thiếu khoảng 19% đều theo hai biến không đủ chứng minh MCAR.
- **Churn temporal definition:** Chưa biết nhãn đại diện cho thời điểm nào; chưa thể kết luận có thể dự đoán tương lai không.
