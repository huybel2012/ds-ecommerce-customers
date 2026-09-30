"""
Đây chỉ là demo kỹ thuật: imputer được fit trên train rồi mới transform test.
Kết quả không được ghi ngược vào dataset cleaned của project.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_ecommerce_raw

REPORT_DIR = PROJECT_ROOT / "reports" / "missing"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    df = load_ecommerce_raw()
    raw_returns = df["Returns"].copy()
    X = df[["Returns"]].copy()

    section("Mục đích")
    print(
        "Chỉ minh họa kỹ thuật: fit imputation trong scikit-learn Pipeline "
        "sau khi đã chia train/test."
    )
    print(
        "Demo này không thay đổi treatment của project: Returns vẫn giữ NaN "
        "trong dữ liệu cleaned và Returns_was_missing vẫn được bảo toàn."
    )

    section("Chia train/test trước khi fit imputer")

    # Stratify theo missingness để train/test có tỷ lệ missing gần nhau.
    missing_strata = X["Returns"].isna().astype("int8")
    stratify = missing_strata if missing_strata.nunique() > 1 else None

    X_train, X_test = train_test_split(
        X,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )

    print(f"Số dòng train: {len(X_train):,}")
    print(f"Số dòng test : {len(X_test):,}")
    print(f"Returns thiếu trong train: {int(X_train['Returns'].isna().sum()):,}")
    print(f"Returns thiếu trong test : {int(X_test['Returns'].isna().sum()):,}")

    section("Pipeline không leakage")

    # Imputer nằm trong Pipeline và chỉ học thống kê từ tập train.
    pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                add_indicator=True,
            ),
        ),
    ])

    X_train_imputed = pipe.fit_transform(X_train)
    X_test_imputed = pipe.transform(X_test)

    imputer = pipe.named_steps["imputer"]
    learned_value = imputer.statistics_[0]
    train_mode = X_train["Returns"].mode(dropna=True).iloc[0]

    print("Imputer chỉ được fit trên TRAIN.")
    print(f"Giá trị Returns học được: {learned_value}")
    print(f"Mode chỉ từ train       : {train_mode}")
    print(f"Kích thước train sau transform: {X_train_imputed.shape}")
    print(f"Kích thước test sau transform : {X_test_imputed.shape}")
    print(f"NaN train sau transform: {int(pd.isna(X_train_imputed).sum()):,}")
    print(f"NaN test sau transform : {int(pd.isna(X_test_imputed).sum()):,}")

    if learned_value != train_mode:
        raise AssertionError("Thống kê của imputer không khớp với mode chỉ tính trên train.")

    section("Kiểm tra toàn vẹn dữ liệu")

    raw_unchanged = raw_returns.equals(df["Returns"])
    raw_returns_missing = int(df["Returns"].isna().sum())

    print(f"Số Returns thiếu trong raw vẫn là: {raw_returns_missing:,}")
    print(f"Returns trong raw không đổi      : {raw_unchanged}")
    print("Không có giá trị imputed nào được ghi ngược vào df.")

    if not raw_unchanged:
        raise AssertionError("Demo đã làm thay đổi Returns trong raw ngoài dự kiến.")

    report = pd.DataFrame([{
        "purpose": "Leakage-safe imputation demonstration",
        "feature": "Returns",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_returns_missing": int(X_train["Returns"].isna().sum()),
        "test_returns_missing": int(X_test["Returns"].isna().sum()),
        "strategy": "most_frequent + add_indicator",
        "fit_scope": "train only",
        "learned_returns_value": learned_value,
        "raw_returns_missing_after_demo": raw_returns_missing,
        "clean_dataset_modified": False,
    }])

    report_path = REPORT_DIR / "leakage_safe_imputation_demo.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu báo cáo: {report_path}")


if __name__ == "__main__":
    main()
