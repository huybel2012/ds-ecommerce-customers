# Datasheet — Synthetic E-commerce Customer Transactions

**Phiên bản phân tích:** Phase 1–7, nhánh `Dung`; 250.000 giao dịch, 49.673 mã khách. **Người chịu trách nhiệm xác minh:** nhóm đồ án (ký duyệt trước nộp). **Trạng thái quyền tái phân phối: CHƯA XÁC MINH.**

## Nguồn gốc & mục đích
- Dataset trên Kaggle: [Shriyash Jagtap — E-commerce Customer Data for Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis); theo mô tả dự án, bản synthetic sử dụng Faker.
- Tệp đang phân tích: `ecommerce_customer_data_custom_ratios.csv`; mô phỏng giao dịch 2020-01-01 đến 2023-09-15.
- Dùng trong **giảng dạy INFO3020** để thực hành data cleaning, EDA, hypothesis tests, regression, classification, fairness. Không thích hợp để quyết định giá, phân bổ ưu đãi, đánh giá tín dụng hay dự đoán churn của khách thật.
- Không thu thập bằng scraping, nên kiểm tra `robots.txt` cho scraping **không áp dụng** với luồng tải Kaggle này. Thay vào đó phải kiểm tra điều khoản dataset Kaggle và quyền tái sử dụng từ tác giả.

## Nội dung, đối tượng & thiếu hụt
- Một dòng đại diện một giao dịch mô phỏng; có nhiều dòng trên `Customer ID`.
- Trường chính: `Purchase Date`, `Product Category`, `Product Price`, `Quantity`, `Total Purchase Amount`, `Payment Method`, `Returns`, `Age`, `Gender`, `Churn`; có `Customer Name` và `Customer ID` giả lập.
- Thiếu `Returns`: 47.596/250.000 (19,0384%); trường này **không** được biến thành 0 trong CSV sạch chuẩn.
- `Total Purchase Amount` khác `Product Price × Quantity` trong 249.952 dòng. Không biết currency, cách tạo cột tiền hoặc có thuế/giảm giá hay không. Hai đại lượng phải được trình bày tách biệt.
- Nhãn `Churn` nhất quán trong từng khách, nhưng **không có event/effective timestamp**. Không biết cohort/cửa sổ quan sát thực tế hoặc horizon dự đoán.

## Đại diện và rủi ro đạo đức
- Nhãn Gender chỉ có hai giá trị `Male`, `Female`; tuổi trong khoảng 18–70. Các nhóm ngoài phạm vi dữ liệu không có cơ sở để đánh giá.
- Vì synthetic và không có thiết kế lấy mẫu từ một tổng thể thật, không thể suy rộng phân phối, hoạt động mua hay công bằng sang khách thật.
- Tên có vẻ giả lập **không đồng nghĩa với chứng nhận ẩn danh**. Trường Customer ID, Gender, Age và chuỗi giao dịch vẫn có tính định danh/quasi-identifier. **Chưa đo k-anonymity**, không công bố mức k đạt được.
- Dữ liệu legacy đã được commit trong Git (`data/processed/ecommerce_customer_cleaned.csv` và `ecommerce_clean.parquet`); `.gitignore` không xóa các bản trong lịch sử.

## Quyền, tính minh bạch và vòng đời
- **License tại metadata Kaggle gốc: chưa được nhóm lưu bằng chứng xác minh.** Không dùng license của Kaggle notebook hoặc GitHub bên thứ ba để thay thế.
- **Consent của người thật:** không áp dụng cho bản tạo giả; mọi dữ liệu thật trong tương lai phải được đánh giá độc lập về quyền riêng tư và căn cứ xử lý.
- Không tự xác nhận quyền phát hành CSV legacy hoặc xử lý lịch sử Git. Tạm giữ trạng thái `NOT_CLEARED_FOR_PUBLIC_DATA_REDISTRIBUTION` cho đến khi có xác nhận từ người phụ trách.
- CSV sạch do Phase 2 tạo cục bộ được ignore; báo cáo công khai chỉ ở cấp tổng hợp, không xuất dự đoán per-customer hoặc bảng tên.

## Tiền xử lý, sử dụng và cập nhật
- `python -m src.run_pipeline` tạo `data/processed/ecommerce_customer_clean.csv`; `python -m src.run_pipeline --validate-only` kiểm tra khớp raw/clean và cờ missing.
- Các phép thử và biểu đồ tái lập theo [Reproducibility Guide](reproducibility_guide.md). Tập dataset không được cập nhật tự động; cần phiên bản, ngày tải và SHA256 riêng khi tái chạy.
- Chưa có chính sách retention chính thức của nguồn Kaggle được xác minh. Nhóm cần quyết định lưu/huỷ file tải cục bộ theo quy định môn học và điều kiện cấp phép thực tế.
