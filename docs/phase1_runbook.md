# Phase 1 Runbook — Data Integrity & Validation

**Nhánh làm việc:** `Dung`. **Giới hạn:** Phase 1 không merge `main`, không ghi đè legacy CSV, không chạy lại cả pipeline cũ đang lỗi.

## 0. Điểm xuất phát đã kiểm tra

- `data/processed/ecommerce_customer_cleaned.csv`: 250.000 dòng × 12 cột; Returns đã bị fill 0, không có missing indicator.
- `data/processed/ecommerce_clean.parquet`: 250.000 dòng, chứa `Returns` với 47.596 NULL (đã kiểm tra metadata Parquet), cùng các cột `Customer Age` và `revenue_calc`.
- `src/pipeline/04_finalize_cleaning.py` đã chứa xử lý giữ NaN cho Returns, nhưng file kết quả CSV đang phát hành cũ không phản ánh quyết định đó. Việc chạy lại script 04 thuộc Phase 2 vì còn thiếu intermediate và utility.

## 1. Chạy đúng trên Windows / PowerShell

```powershell
cd C:\Code\KHDL\Project
git switch Dung
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.phase1_data_integrity
```

Venv của nhóm đã có `pandas` và `pyarrow`. Nếu file Parquet không tồn tại hoặc không khớp với CSV, script **dừng có lỗi**. Không được sửa thủ công để bỏ qua kiểm tra.

### File kết quả

- `data/processed/ecommerce_customer_integrity.csv`: **bản mới**, giữ NULL Returns, thêm `Returns_was_missing` và `line_item_value_calc`, không có `Customer Name`.
- `reports/phase1/data_integrity_metrics.json`: số liệu JSON kiểm toán trước/sau.
- `reports/phase1/data_integrity_report.md`: giải thích sáu chiều chất lượng và các khuyến nghị diễn giải.

Script không ghi đè `ecommerce_customer_cleaned.csv`, `ecommerce_clean.parquet`, biểu đồ hay `app.py`.

## 2. Kiểm tra đầu ra

```powershell
.\venv\Scripts\python.exe -c "import pandas as pd; p='data/processed/ecommerce_customer_integrity.csv'; d=pd.read_csv(p); print('Rows:',len(d)); print('Missing Returns:',d['Returns'].isna().sum()); print('Missing indicator:',d['Returns_was_missing'].sum()); print('Return rate observed:',d['Returns'].mean()); print('Amount totals:',d['Total Purchase Amount'].sum(),d['line_item_value_calc'].sum())"
```

Đối với phiên bản đã kiểm toán, kỳ vọng: 250.000 dòng; 47.596 Returns missing; indicator tổng 47.596; tỷ lệ trả hàng trên dòng quan sát ~49,79%; hai tổng tiền là 681.342.683 và 190.908.939.

Nếu các số khác, xem report trước khi commit.

## 3. Không làm trong Phase 1

- **Không** tự đổi `Total Purchase Amount` thành `Price × Quantity`.
- **Không** đổi dữ liệu của dashboard ngay; Dashboard và các mô hình vẫn đang đọc `ecommerce_customer_cleaned.csv`. Đồng bộ phía hiển thị, metric và kiểm định sẽ thực hiện ở Phase 3/4 sau khi Phase 2 tái lập pipeline.
- **Không** commit file Parquet hoặc CSV raw có thể bị ràng buộc giấy phép, và không commit bản CSV output khoảng 20MB vào repo khi chưa thống nhất chính sách data.
- **Không** khẳng định giấy phép Kaggle hoặc đơn vị tiền khi chưa được xác minh trực tiếp.

## 4. Cập nhật nhánh Dung sau khi chạy

```powershell
git status
git add .gitignore src/phase1_data_integrity.py tests/test_phase1_data_integrity.py docs/data_dictionary.md docs/phase1_runbook.md reports/phase1/data_integrity_report.md reports/phase1/data_integrity_metrics.json reports/phase1/phase1_findings.md
git commit -m "feat(data): audit integrity and recover missing Returns safely"
git push origin Dung
```

Không dùng `git add .` nếu có CSV đầu ra/bản dữ liệu chưa được phép đăng công khai. Đây là hướng dẫn thao tác, không đồng nghĩa các file đã được commit.
