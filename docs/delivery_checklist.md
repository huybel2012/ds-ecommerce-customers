# Phase 7 — Delivery Checklist & Human Sign-off

**Scope:** #23 documentation/reproducibility, #25 final report/defense, #26 final repository audit. **No PR or merge**. `merge deferred` by user; `main` remains untouched by this phase.

## A. Technical review (automated/read-only)

- [x] README có nguồn dữ liệu, chạy lại, liên kết reports và **AI assistance**.
- [x] Có báo cáo cuối kỳ `reports/final_project_report.md` với kết quả truy vết W1–W7, W9–W12.
- [x] Có datasheet, defense guide, reproducibility guide và model card.
- [x] Unit tests được chạy trong môi trường xây dựng patch; **máy nhóm phải nghiệm thu lại**.
- [x] Audit Phase 7 kiểm tra kết quả đã commit, 14 hình và tính nhất quán frozen model.
- [x] Không có code thay đổi threshold, labels, training split, kết quả W9–W12 hoặc `main`.

## B. Human gates trước nộp và trước công bố công khai

- [ ] Xác minh **giấy phép của chính Kaggle dataset** từ nguồn gốc, lưu ngày/đường dẫn/bằng chứng và quyền tái phân phối.
- [ ] Quyết định xử lý hai file legacy **đã commit**: `data/processed/ecommerce_customer_cleaned.csv` và `data/processed/ecommerce_clean.parquet` (chứa synthetic names/ID). Tính khả dụng Git history cần được rà soát; `.gitignore` không xóa history. **Không tự rewrite Git history trong Phase 7.**
- [ ] Kiểm tra chính sách dữ liệu của môn học/nhà trường, retention và điều kiện công bố public; nguồn fake không tự cấp phép hay chứng nhận ẩn danh.
- [ ] Đối chiếu phần AI disclosure với thực tế các thành viên đã sử dụng công cụ; ký xác nhận mã/biểu đồ/bản thảo đã được kiểm tra bởi người.
- [ ] Bổ sung tên nhóm, MSSV, phân công và bằng chứng đóng góp thật. Không tạo tên/mã sinh viên giả.
- [ ] Xem báo cáo cuối, đảm bảo không thêm business/currency/causal/fairness claims không được data hỗ trợ.
- [ ] Xác nhận model card **một trang** khi xuất bản nộp, chuẩn trình bày theo hướng dẫn giảng viên.
- [ ] Kiểm tra nội dung và quy cách file nộp (PDF/slides/code/demo) theo thông báo giảng viên; không suy đoán deadline/độ dài từ README.
- [ ] Xác nhận website/dashboard chạy trên máy demo sử dụng **canonical Phase 2 clean CSV**, không fallback sang legacy.

## C. Final reproduction trên máy Windows của nhóm

```powershell
cd C:\Code\KHDL\Project
git switch Dung
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.run_phase7 --dry-run
.\venv\Scripts\python.exe -m src.run_phase7
git status
```

Đối chiếu `reports/phase7/release_audit.md`/`.json`. Audit phải báo `Technical readiness: PASS` và **`Public redistribution: NOT CLEARED`** nếu chưa xử lý gate license. Sau nghiệm thu trên Windows, chỉ commit các docs/tests/audit Phase 7; **không** commit raw CSV, local env, cache hoặc các manifest có đường dẫn cá nhân.

## D. Quyết định repo/nhánh

- [x] Baseline Phase 6 trên `Dung`: commit `2ab6bae`.
- [x] Scope Phase 7 không gồm merge, không tạo PR.
- [ ] Nhóm review diff Phase 7, chạy `git diff --cached --check`, commit/push `Dung` sau nghiệm thu.
- [ ] Chỉ xem xét PR/merge về sau **khi người dùng/chủ repo yêu cầu riêng**; không mặc định nhánh `main` đã được phát hành.

**Final status:** Handover tài liệu và kỹ thuật có thể sẵn sàng sau nghiệm thu; **public dataset release và phân công học thuật vẫn chờ xác nhận của con người**.
