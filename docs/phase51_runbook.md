# Phase 5.1: Classification Improvement Study

**Trạng thái:** thăm dò bổ sung, không thay Phase 5, không thay W11 chính thức hoặc đề xuất triển khai. Nhánh `Dung` sau commit `1b0e51a`; tương thích Python 3.11 và dependencies đã cài ở Phase 2.

## Câu hỏi nghiên cứu

Với nhãn Churn tĩnh mô phỏng không có event timestamp, liệu những đặc trưng hành vi có tạo ra **khả năng phân biệt thực sự vượt mức ngẫu nhiên**, hay F1 cao chỉ do dự đoán toàn bộ khách là Churn?

### Protocol đã xác định trước khi xem kết quả Phase 5.1

1. **Đầu vào duy nhất:** `data/processed/ecommerce_customer_clean.csv` từ Phase 2, `Returns=NULL` và `Returns_was_missing` khớp. Không dùng CSV legacy hoặc tự điền missing. Mã chỉ đọc, không ghi đè dataset.
2. **Cutoff:** mọi feature chỉ từ `Purchase Date < 2023-01-01`. Nhãn Churn tĩnh không có thời điểm xác lập nên tuyệt đối không gọi đây là dự đoán churn tương lai.
3. **Đơn vị:** một khách một dòng; dùng `Customer ID` làm group/index, không phải predictor; không dùng `Churn`, `Returns`, missing flags, tuổi, giới tính, tên khách và bất kỳ giao dịch nào sau cutoff làm features.
4. **Tập chọn mô hình:** giữ đúng split GroupShuffleSplit(seed=42) của Phase 5; chỉ dùng **39.302 khách train** cho các nghiên cứu. 9.826 khách holdout của Phase 5 **không được chấm lại** ở Phase 5.1, vì kết quả Phase 5 đã được xem và không còn là test độc lập mới.
5. **Feature:** 19 feature Phase 5 so với 28 (19+9) feature. Chín biến mới: khoảng cách mua trung bình/độ lệch, số giao dịch 90/180 ngày, số giao dịch giai đoạn 90–180 ngày trước cutoff, thay đổi tần suất 90 ngày, tần suất đổi danh mục, xu hướng giá trị ghi nhận 180 ngày, cờ có đủ 2 cửa sổ 180 ngày. Giá trị ghi nhận *không* được coi là USD/doanh thu thực. Với các khách có một giao dịch, gap được gán 0 và cờ giai đoạn biểu thị thiếu lịch sử.
6. **Mô hình:** Logistic Regression và `HistGradientBoostingClassifier` (tham số định trước). 4 nhánh 2 tập features × 2 mô hình. Repeated grouped stratified CV: mặc định 2 seed, 3 fold mỗi seed. ROC-AUC dùng để sàng lọc, AP/PR-AUC và Lift@Top-10% để kiểm tra tính hữu ích. Không tối ưu threshold F1 trên holdout.
7. **Đối chứng âm:** sau khi chọn *ứng viên thăm dò tốt nhất trên train*, hoán vị nhãn của cùng train customers, chạy grouped CV bằng mô hình tương ứng với 2 lần xáo trộn độc lập. Kết quả chỉ ước lượng khả năng khớp nhiễu, không chứng minh nhãn được tác giả tạo ngẫu nhiên.
8. **Univariate audit:** AUC từng feature trên train; mutual information kNN trên mẫu ngẫu nhiên 5.000 *khách TRAIN*, vì kNN MI trên toàn bộ 40.000 khách tốn nhiều thời gian và có thể nhiễu ở biến rời rạc. Không nhìn test.

### Ngưỡng sàng lọc bảo thủ (không phải test ý nghĩa thống kê)

Muốn *đề xuất thu thập dữ liệu xác nhận mới*, tất cả điều kiện TRAIN CV phải thỏa:

- ROC-AUC trung bình ≥ **0.55**;
- Average precision - prevalence trung bình ≥ **0.03**;
- Lift@Top-10% trung bình ≥ **1.20**;
- ROC-AUC trung bình cao hơn control nhãn đảo ≥ **0.03**;
- ROC-AUC trung bình từng CV repeat ≥ **0.52**.

Không tiêu chí nào đồng nghĩa được triển khai; do đã thử 4 cấu hình và chọn cấu hình cao nhất, có *multiple-testing / winner's curse*. Nếu không vượt gate, **giữ nguyên Phase 5** và báo cáo tín hiệu phân biệt chưa đủ. Kể cả PASS, cần nhãn có mốc thời gian và tập dữ liệu xác nhận độc lập trước khi thay đổi báo cáo W11.

## Cách chạy trên Windows

```powershell
cd C:\Code\KHDL\Project
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.run_phase51 --dry-run
.\venv\Scripts\python.exe -m src.run_phase51
```

Chạy chính thức không thêm `--folds`, `--repeats`, `--permutations` để dùng đầy đủ protocol 3-fold × 2 repeats và 2 permutations. Có thể kiểm tra nhanh bằng `--folds 2 --repeats 1 --permutations 1 --max-iter 20`, nhưng **không dùng kết quả quick run làm kết luận cuối**.

**Các output được tạo (4 tracked artifacts):**

- `reports/phase5/phase51_improvement_study.md` (báo cáo train-only)
- `reports/phase5/phase51_metrics.json` (metrics và từng CV fold)
- `reports/figures/13_phase51_univariate_signal.png`
- `reports/figures/14_phase51_repeated_cv.png`

Manifest `reports/phase5/phase51_run_manifest.json` chứa đường dẫn máy cá nhân, timestamp, SHA256: được `.gitignore` loại trừ. Các file Phase 1–5 vẫn nguyên vẹn, đặc biệt `reports/classification_results.md` và `reports/phase5/classification_metrics.json`.

## Cần kiểm tra khi nghiệm thu

- 82 bài kiểm thử (67 cũ + 15 mới) thành công; Python 3.11 không có lỗi syntax/import.
- Pipeline đầu vào raw→clean validation PASS; không sửa file dữ liệu sạch.
- Báo cáo chỉ so sánh trong tập train; không gọi synthetic label là future churn.
- `git status`: chỉ chứa source, docs, workflow và bốn artifacts Phase 5.1 mới. Không đưa `data/raw`, `venv` hay CSV processed lên GitHub.
- **Kết quả Phase 5 không bị ghi đè.** Chỉ commit sau nghiệm thu tại Windows.
