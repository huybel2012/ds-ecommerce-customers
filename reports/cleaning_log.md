# Cleaning Log

**Dataset:** `data/raw/ecommerce_customer_data_custom_ratios.csv` (250,000 dòng, 13 cột lúc đầu)
**Notebook:** `01_data_cleaning.ipynb`
**Nguyên tắc:** file raw giữ nguyên, mọi thay đổi chỉ làm trên `df` trong notebook.

| Step | Column | Finding | Action | Rows | Why |
|---|---|---|---|---|---|
| 3.1 | Customer Age | Giống hệt cột `Age` (`(df["Customer Age"] == df["Age"]).all()` = True) | Drop `Customer Age`, giữ `Age` | 250,000 | Cột trùng thông tin, không mất dữ liệu |
| 3.2 | Returns | 19.0% NaN (47,596 dòng); tỉ lệ thiếu đều theo Payment Method và Product Category (~19% mỗi nhóm) | Fill 0 + tạo cột `Returns_was_missing` | 47,596 | Giả định MCAR dựa trên 2 cột đã kiểm tra. Giữ indicator để không mất dấu vết. Tỉ lệ trả hàng trên dòng quan sát được là 0.498, sau khi fill còn ~0.40, nên khi báo cáo tỉ lệ trả hàng chỉ tính trên dòng `Returns_was_missing == 0` |
| 4.1 | Age | hist(bins=50) cho thấy 3 đỉnh bất thường ở ~20, ~44, ~70. Điều tra: tuổi "đỉnh" cao nhất chỉ chiếm 5,009/250,000 dòng (~6% trên trung bình), Customer ID nhóm này trải đều toàn khoảng (10–49,974), không co cụm | Giữ nguyên | 0 | Kiểm chứng lại bằng `bins=range(18,72)` khớp đúng 53 giá trị tuổi nguyên → phân phối phẳng đều, 3 đỉnh biến mất. Kết luận: binning artifact, không phải outlier thật |
| 4.2 | Product Price, Total Purchase Amount | Modified Z-score (\|M\| > 3.5) và IQR (1.5×) đều gắn cờ 0 dòng | Giữ nguyên (Keep) | 0 | Phân phối gần đều (Price 10–500), không có điểm cực đoan để điều tra |
| 4.3 | Total Purchase Amount | Khác `Product Price × Quantity` ở 249,952 dòng (99.98%); chênh lệch từ -2,330 đến +5,314; corr = -0.0017 | Giữ nguyên, không sửa. Khi cần dùng thì tạo cột riêng `revenue_calc = Price × Quantity` | 249,952 | Sai lệch ở gần như mọi dòng và corr ≈ 0 nên không phải lỗi nhập từng dòng, nhiều khả năng Total được sinh độc lập. Ghi đè sẽ làm mất dữ liệu gốc mà không có bằng chứng |
| 4.4 | Customer ID | Mỗi ID chỉ có 1 giá trị Name, Age, Gender, Churn (0 ID mâu thuẫn) | Giữ nguyên | 0 | Kiểm tra bằng `groupby("Customer ID").nunique()` |
| 4.5 | Duplicates | 0 dòng trùng hoàn toàn; 0 dòng trùng theo key (Customer ID, Purchase Date) | Giữ nguyên | 0 | `df.duplicated()` |
| 4.6 | Purchase Date | Dạng chuỗi | Parse sang datetime | 250,000 | 0 giá trị lỗi, khoảng 2020-01-01 đến 2023-09-15, không có ngày tương lai |
| 4.7 | Age, Quantity, Product Price, Purchase Date | 4 validity rules đều đạt: Age 18–100, Quantity ≥ 1, Price > 0, ngày không ở tương lai | Giữ nguyên | 0 | Chạy dict `checks` |
| 5.1 | Product Category, Payment Method, Gender | 3 cột categorical, cardinality thấp (4, 4, 2 levels) | One-hot encode (`drop_first=True`), đưa vào `ColumnTransformer` | 250,000 | Không có thứ tự thật giữa các levels; cardinality thấp nên không cần group "Other" hay frequency encoding |
| 5.2 | Product Price, Quantity, Total Purchase Amount, Age | 4 cột số, không có outlier (xác nhận ở 4.2) | `StandardScaler` trong Pipeline (fit trên train only khi modeling) | 250,000 | Không cần RobustScaler vì không giữ outlier nào; Standard là default cho PCA/SVM/regression |
| 5.3 | (4 cột số) | Ma trận tương quan: hệ số cao nhất 0.052 (Total vs Age) | Không drop cột nào | 0 | Không có cặp nào > 0.95, không có thông tin dư thừa cần loại bỏ |
| 5.4 | — | Pipeline test: `prep.fit_transform(X)` chạy OK, output `(250000, 11)` | Lưu `data/processed/ecommerce_clean.parquet` (bản clean, chưa scale/encode) | 250,000 | Scale/encode phải fit trên train only khi modeling (Week 10-11) để tránh leakage — không fit sẵn rồi lưu file |


## Ghi chú về leakage

- Các bước 3.1, 3.2, 4.5 là sửa lỗi hoặc đổi kiểu, không học tham số từ dữ liệu, nên làm trước train/test split được.
- Bước 5.1, 5.2 (encoding, scaling) KHÔNG được fit trên toàn bộ `df` rồi lưu ra file — phải nằm trong `Pipeline`, fit lúc `model.fit(X_train, y_train)` ở Week 10-11. File `processed` chỉ chứa bản đã clean (missing/outlier/consistency), chưa qua encode/scale.
- Fill Returns bằng hằng số 0 không tính thống kê từ dữ liệu, nên không gây leakage.
- Nếu sau này model Churn: mỗi khách có nhiều dòng, nên split theo `Customer ID` (`GroupShuffleSplit`), tránh một khách nằm ở cả train lẫn test.