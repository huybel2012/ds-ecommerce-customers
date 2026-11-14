# Reproducibility Guide — INFO3020 Final Delivery

**Đối tượng:** thành viên nhóm, giảng viên kiểm thử trên Windows PowerShell, Python 3.11; **nhánh:** `Dung`. Không đòi hỏi Git merge. Không dùng dữ liệu legacy như một bản thay thế cho cleaned chuẩn.

## 1. Chuẩn bị môi trường

```powershell
# Thực hiện sau khi clone/checkout nhánh Dung
cd C:\Code\KHDL\Project
git switch Dung
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip freeze > environment_local.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

`environment_local.txt` là bằng chứng phiên bản trên máy, **không tự ý commit** vì đường dẫn và dependency có thể thay đổi; cần chọn nơi lưu riêng/đưa vào hồ sơ bàn giao. `requirements.txt` đặt range phụ thuộc chứ chưa lock hoàn toàn; các metrics có thể xê dịch nhẹ theo phiên bản (đặc biệt ranking AUC/AP). Báo cáo Phase 6 so lại confusion matrix với Phase 5 và ghi delta để kiểm soát drift.

## 2. Bổ sung raw dataset (bắt buộc)

- Mở [Kaggle dataset của Shriyash Jagtap](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis), kiểm tra provenance/license của chính dataset và tải file `ecommerce_customer_data_custom_ratios.csv`.
- Đặt tại `data/raw/ecommerce_customer_data_custom_ratios.csv`; không commit file raw.
- **Không** dùng `data/processed/ecommerce_customer_cleaned.csv` hay `data/processed/ecommerce_clean.parquet` (legacy). Phase 2 và các Phase 3–6 chỉ dùng `data/processed/ecommerce_customer_clean.csv` do pipeline tạo.

## 3. Thứ tự chạy có kiểm soát

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline --dry-run
.\venv\Scripts\python.exe -m src.run_pipeline
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.pipeline.05_explanatory_visualization
.\venv\Scripts\python.exe -m src.run_phase4 --dry-run
.\venv\Scripts\python.exe -m src.run_phase4
.\venv\Scripts\python.exe -m src.run_phase5 --dry-run
.\venv\Scripts\python.exe -m src.run_phase5
.\venv\Scripts\python.exe -m src.run_phase51 --dry-run
.\venv\Scripts\python.exe -m src.run_phase51
.\venv\Scripts\python.exe -m src.run_phase6 --dry-run
.\venv\Scripts\python.exe -m src.run_phase6
.\venv\Scripts\python.exe -m src.run_phase7 --dry-run
.\venv\Scripts\python.exe -m src.run_phase7
```

Bước Phase 5.1 thử thêm đặc trưng **trên train only**, không đánh giá lại test Phase 5; Phase 6 chạy lại mô hình đã được chốt và so sánh frozen metrics, không train lại để chọn threshold. Phase 7 **chỉ đọc các báo cáo/metrics đã commit**, kiểm toán tính nhất quán và viết hai audit files, không đọc raw/clean hay xác nhận luật bản quyền.

**Dashboard:**

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Không chạy các script pipeline lẻ không nằm trong thứ tự trên như một bằng chứng độc lập của source-of-truth. Đồ án có các notebook tham khảo, còn các lệnh `python -m` trên mới là quy trình tái lập ưu tiên.

## 4. Bảng xác minh kết quả

| Mốc | Check sau khi chạy | Output |
|---|---|---|
| Unit tests | `Ran ... tests`, `OK` | Không cần Kaggle raw |
| Phase 2 | `PASS: raw -> cleaned integrity gates` | `data/processed/ecommerce_customer_clean.csv`, `reports/phase2/run_manifest.json` |
| Phase 3 | Hình 01–05 | `reports/figures/01_...` đến `05_...` |
| W9 | 3 giả thuyết và Holm-adjusted p | `reports/hypothesis_tests.md`, hình 09, `reports/phase4/hypothesis_results.json` |
| W10 | Group holdout, baseline MAE | `reports/regression_results.md`, hình 06–08, `reports/phase4/regression_metrics.json` |
| W11 | Holdout 9.826 khách, không overlap; ROC-AUC≈0,495 | `reports/classification_results.md`, hình 10–12, `reports/phase5/classification_metrics.json` |
| Phase 5.1 | Signal screen FAIL (không phải lỗi phần mềm) | `reports/phase5/phase51_improvement_study.md`, hình 13–14 |
| W12 | `PASS: Phase 6, frozen ... verified` | `reports/phase6/ethics_and_bias.md`, `model_card.md`, `fairness_metrics.json` |
| Final audit | `Technical readiness: PASS` và `Public redistribution: NOT CLEARED` | `reports/phase7/release_audit.md`, `.json` |

Khi khác biệt vượt ngưỡng hoặc file thiếu, **không sửa số trong báo cáo bằng tay**. Kiểm tra SHA raw/clean, version package, tiền xử lý, seed; tái chạy pipeline và xem log lỗi. `Reports/` chứa các kết quả đã chạy trước đó; chỉ tuyên bố tái lập trên máy mới sau khi các lệnh thực sự được chạy.

## 5. Evidence & quyền riêng tư

`reports/phase2/run_manifest.json`, `reports/phase4/run_manifest.json`, `reports/phase5/run_manifest.json`, `reports/phase5/phase51_run_manifest.json`, `reports/phase6/run_manifest.json` là metadata cục bộ, có đường dẫn/thời gian/hash. Không upload nguyên file lên nơi công khai khi chưa rà soát. Các hình và JSON report đã commit chứa **số liệu tổng hợp**, không xuất Customer Name/ID.

Dataset license, quyền tái phân phối, nhóm tên giả lập trong các tệp legacy và thông tin người thực hiện là **manual gates**. Sự thành công của pipeline không tự thông qua các gate này.

## 6. Bản nộp cho môn học

- Tài liệu chính: `reports/final_project_report.md` (cần người biên tập/giảng viên duyệt định dạng trước khi xuất PDF), `reports/phase6/model_card.md` một trang, `docs/dataset_datasheet.md` và `docs/ai_usage_disclosure.md`.
- Thuyết trình: dùng `docs/defense_guide.md`, tự điền thành viên/đóng góp đã được xác nhận và chuẩn bị demo dashboard.
- Kiểm tra mục cuối: `docs/delivery_checklist.md` và output `python -m src.run_phase7`.
- Không bao gồm W8 giữa kỳ, bài EX hoặc merge vào `main` trong bàn giao hiện tại.
