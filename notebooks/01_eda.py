import os
import pandas as pd
import numpy as np

#thiet lap duong dan va thu muc
RAW_DATA_PATH = "data/raw/ecommerce_customer_data_custom_ratios.csv"
PROCESSED_DATA_PATH = "data/processed/ecommerce_customer_cleaned.csv"
REPORTS_DIR = "reports"
CLEANING_LOG_PATH = os.path.join(REPORTS_DIR, "cleaning_log.csv")

os.makedirs("data/processed", exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

#danh sach luu tru nhat ki
cleaning_logs = []

def record_log(step_name, reason, rows_before, rows_after, affected_count):
    """Ghi nhận lại từng bước biến đổi dữ liệu"""
    cleaning_logs.append({
        "buoc_xu_ly": step_name,
        "ly_do": reason,
        "so_dong_truoc": rows_before,
        "so_dong_sau": rows_after,
        "so_dong_bi_anh_huong": affected_count
    })

print("=" * 50)
print("Nạp dữ liệu từ thư mục raw")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Kích thước ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")


#Xu li trung lap ban ghi
print("\nKiểm tra bản ghi trùng lặp")
rows_before = len(df)
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
rows_after = len(df)

record_log(
    step_name="Loai bo trung lap hoan toan",
    reason="Loai bo cac dong du lieu trung lap toan bo thong tin",
    rows_before=rows_before,
    rows_after=rows_after,
    affected_count=dup_count
)
print(f"Đã loại bỏ {dup_count} dòng trùng lặp.")

#xu li gia tri thieu o cot return
print("\nXử lý giá trị khuyết thiếu (NaN) tại cột Returns...")
rows_before = len(df)
missing_returns = df["Returns"].isna().sum()

#dien 0 cho cac don vi khong hoan hang, ep kieu ve int
df["Returns"] = df["Returns"].fillna(0).astype(int)
rows_after = len(df)

record_log(
    step_name="Xu ly missing value cot Returns",
    reason="Gia tri NaN duoc quy uoc la don hang thanh cong, khong co yeu cau tra hang (dien 0)",
    rows_before=rows_before,
    rows_after=rows_after,
    affected_count=missing_returns
)
print(f"Đã điền 0 cho {missing_returns} giá trị rỗng tại cột Returns.")


# chuan hoa du lieu ngay thang
print("Chuyển đổi định dạng ngày tháng (Purchase Date)")
rows_before = len(df)
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")

#Loai bo neu khong doc duoc ngay thang
invalid_dates = df["Purchase Date"].isna().sum()
if invalid_dates > 0:
    df = df.dropna(subset=["Purchase Date"])
rows_after = len(df)

record_log(
    step_name="Chuan hoa Purchase Date",
    reason="Ep kieu tu chuoi (str) sang datetime64 de phuc vu phan tich chuoi thoi gian",
    rows_before=rows_before,
    rows_after=rows_after,
    affected_count=rows_before if invalid_dates == 0 else invalid_dates
)
print("Đã chuyển đổi cột Purchase Date sang kiểu datetime.")

#xu li cot trung lap du lieu (custome age va age)
print("Loại bỏ cột trùng lặp thông tin...")
rows_before = len(df)
if "Customer Age" in df.columns and "Age" in df.columns:
    df = df.drop(columns=["Customer Age"])
    affected = rows_before
    reason_desc = "Xoa cot Customer Age do trung lap nghiep vu hoan toan voi cot Age"
else:
    affected = 0
    reason_desc = "Khong phat hien cot du thua can xoa"

rows_after = len(df)
record_log(
    step_name="Xoa cot du thua Customer Age",
    reason=reason_desc,
    rows_before=rows_before,
    rows_after=rows_after,
    affected_count=affected
)
print("Đã xóa cột Customer Age.")


# xuat file ket qua va nhat ki lam sach
print("\nĐang lưu dữ liệu đã xử lý và xuất cleaning_log")

# xuat nhat ki lam sach reports/cleaning_log.csv
log_df = pd.DataFrame(cleaning_logs)
log_df.to_csv(CLEANING_LOG_PATH, index=False, encoding="utf-8-sig")
print(f"Nhật ký đã được lưu tại:{CLEANING_LOG_PATH}")

#xuat du lieu sau chuan hoa vao data/processed/ecommerce_customer_cleaned.csv
df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
print(f"Dữ liệu sạch đã lưu tại: {PROCESSED_DATA_PATH}")

print("=" * 50)
print("HOÀN TẤT BƯỚC TIỀN XỬ LÝ DỮ LIỆU.")