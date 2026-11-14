# Hướng dẫn bảo vệ đồ án INFO3020 — Phase 7

**Mục tiêu:** giúp thành viên giải thích được từng bước đã thực thi, giả định và kết quả âm tính. Đây là **dàn ý gợi ý** cho một buổi trình bày khoảng 10–12 phút; thời lượng chính thức theo giảng viên. Không dựng số, không coi kết quả synthetic là ứng dụng thực tế. Bỏ W8 và các bài EX.

## A. Dàn ý 10 slide (gợi ý)

| Slide | Trọng tâm | Hình/số nên chiếu | Câu nói phải đúng |
|---|---|---|---|
| 1 | Bài toán và phạm vi | Title, repo `Dung` | Học phần thực hành Data Science trên dữ liệu mô phỏng |
| 2 | Nguồn dữ liệu và unit | 250.000 giao dịch, 49.673 khách | Mỗi khách có nhiều giao dịch; không giả định dữ liệu thật |
| 3 | Data quality & cleaning | 47.596 missing Returns; 249.952 amount mismatch | Không điền missing = 0; amount chưa xác minh |
| 4 | EDA/dashboard | Hình 01 + 03 + 05 | Churn tính theo khách; tháng 09/2023 không hoàn chỉnh |
| 5 | W9: kiểm định | Hình 09, Holm p 0,240/0,624/0,167 | Không bác bỏ H0; đây là liên hệ chứ không phải causal |
| 6 | W10: regression | Hình 08, R²≈0,0032 | MAE tốt hơn baseline rất nhỏ, không có tín hiệu mạnh |
| 7 | W11: classification | Hình 10–11; ROC-AUC≈0,495 | F1≈0,329 bằng all-positive baseline; không dùng thực tế |
| 8 | Phase 5.1: nghiên cứu âm tính | Hình 14; CV best AUC≈0,499, shuffled≈0,503 | Thêm feature/algorithm không tạo cải thiện ổn định |
| 9 | W12: fairness, privacy, scale | Confusion matrix, gender/age gaps, model card | Parity gần như đoán hết dương không phải fairness; license còn mở |
| 10 | Đóng góp, giới hạn, hỏi đáp | Summary, repo và audit | Phương pháp được tái lập; không merge main, không tuyên bố phát hành data |

Mỗi hình đều có trong `reports/figures/`. Không cắt bỏ chú thích chỉ để biểu đồ đẹp. Đừng biến các giả thuyết W9 thành một thực nghiệm A/B đã diễn ra.

## B. Demo trực tiếp (kiểm tra trước trên máy nhóm)

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.run_phase7 --dry-run
.\venv\Scripts\python.exe -m streamlit run app.py
```

Khi mở dashboard: chỉ ra 250k giao dịch, tỷ lệ Churn per-customer, tỷ lệ observed Returns và tháng cuối không trọn vẹn. Nếu không có Internet tại buổi bảo vệ, dữ liệu Kaggle đã tải **hợp lệ** phải có trên máy trình diễn, không lấy legacy CSV để âm thầm thay thế.

## C. 15 câu hỏi phản biện và ý trả lời

1. **Tại sao chọn dataset synthetic?** Để hoàn tất chu trình W1–W12 với số giao dịch đủ lớn và kiểm chứng chất lượng; không đại diện hành vi thật.
2. **Vì sao 250k dòng nhưng chỉ 49.673 khách?** Một khách có nhiều giao dịch. Churn và độc lập thống kê phải xét theo khách.
3. **Vì sao không thay Returns NA thành 0?** NA là chưa quan sát, không có bằng chứng là không trả hàng; làm vậy hạ tỷ lệ observed Returns từ 49,79% xuống 40,31%.
4. **Hai cột tiền khác nhau do thuế hay giảm giá?** Không có chứng cứ metadata; không được tự gán nguyên nhân. Tách reported amount và calculated line value.
5. **Tại sao Books dẫn đầu?** Tổng recorded amount theo danh mục trong sample là lớn nhất (204.939.601), không chứng minh dẫn đầu thị trường.
6. **Kiểm định H1 có p = 0,12 nghĩa là hai giới bằng nhau?** Không. Nó chỉ là chưa đủ chứng cứ bác bỏ H0 dưới giả định kiểm định.
7. **Vì sao dùng Holm?** Kiểm soát family-wise error khi có ba giả thuyết chính; tránh cherry-pick một p nhỏ.
8. **H3 có một so sánh pairwise Holm p ≈ 0,045?** Thăm dò family cặp riêng; omnibus trong family ba kiểm định chính đã điều chỉnh p≈0,167. Không khẳng định phát hiện chắc chắn hay tác động kinh tế.
9. **A/B có chạy thật không?** Chưa. Không có random assignment; báo cáo là bản thiết kế thử nghiệm.
10. **R² = 0,0032 nói gì?** Mô hình giải thích khoảng 0,32% biến thiên của mục tiêu trên test theo định nghĩa R²; tín hiệu rất yếu, đặc biệt target amount còn chưa rõ nghĩa.
11. **Vì sao classifier F1 = 0,329 nhưng không có giá trị?** All-positive baseline cũng ≈0,329, model gần như gọi mọi khách Churn, false positive 7.886/9.826 test cases.
12. **Đã ngăn leakage thế nào?** Features trước 2023-01-01, một dòng/khách, holdout/group CV tách Customer ID; imputer/scaler fit trong CV, threshold chỉ chọn từ train OOF.
13. **Tại sao thêm 9 features mà ROC-AUC không tăng?** Train-only repeated CV và shuffled-label negative control đều gần chance; không đủ tín hiệu, nên giữ kết quả âm tính.
14. **Khoảng cách selection-rate theo giới nhỏ có chứng minh fairness không?** Không. Tất cả nhóm đều gần như bị dự đoán Churn, vì vậy parity không đại diện hiệu quả hoặc công bằng cho người thật.
15. **Nhóm đã làm gì với quyền riêng tư và AI?** Đã công khai nguồn synthetic/AI assistance, ghi các gate về license và legacy names; không nói đã ẩn danh, có k, hay đã được phép phát hành dữ liệu khi chưa chứng minh.

## D. Thực hành theo vai trò, không gán công lao giả

**Nếu nhóm có ba thành viên**, cân nhắc chia: (1) nguồn dữ liệu/cleaning/EDA, (2) thống kê W9–W10, (3) W11–W12, kết luận và demo. Đây chỉ là **đề xuất trình bày**, không phải bản xác nhận ai đã làm phần nào. Trước buổi bảo vệ, mỗi người phải giải thích được mã và kết quả họ giới thiệu, đồng thời nhóm xác nhận phân công theo thực tế và lịch sử commit.

## E. Những câu tuyệt đối không nói

- "Dự đoán khách sắp rời bỏ chính xác" hoặc "ROC-AUC tốt".
- "Đã chạy A/B và chứng minh quan hệ nhân quả".
- "Khách nữ/nam được đối xử công bằng" chỉ dựa vào cùng tỷ lệ dự đoán.
- "Dữ liệu đã ẩn danh 100%, license CC0" nếu chưa có bằng chứng.
- "Doanh thu bằng 681 triệu USD/VND" khi currency/semantics không rõ.

**Chuẩn bị cuối:** chạy checklist tại `docs/delivery_checklist.md`, xem audit, rà soát các nguồn slide thật và xác nhận AI disclosure.
