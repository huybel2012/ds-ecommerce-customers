import pandas as pd
import numpy as np
import unicodedata

# ============ W3 — Audit & Missing ============

HIDDEN_MISSING_TOKENS = {
    "", "-", "--", "?", "n/a", "na", "null", "none",
    "unknown", "missing", "nan", "không rõ", "thỏa thuận", "negotiable",
}

def audit(df: pd.DataFrame) -> pd.DataFrame:
    """One-glance quality report — dtype, missing, unique, sample."""
    sample = df.iloc[0] if not df.empty else pd.Series([pd.NA] * len(df.columns), index=df.columns)
    return pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
        "sample": sample,
    })


def detect_hidden_missing(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Tìm missing ngụy trang (N/A, -, không rõ...) trong cột text.
    Trả về DataFrame đếm số dòng bị ẩn theo từng cột."""
    columns = columns or df.select_dtypes(include=["object", "string"]).columns
    result = {}
    for c in columns:
        normalized = df[c].astype("string").str.strip().str.casefold()
        mask = normalized.isin(HIDDEN_MISSING_TOKENS) & df[c].notna()
        if mask.sum() > 0:
            result[c] = mask.sum()
    return pd.Series(result, name="n_hidden_missing").sort_values(ascending=False)


def missing_by_group(df: pd.DataFrame, target_col: str, group_col: str) -> pd.Series:
    """Tỉ lệ missing của target_col theo nhóm group_col — phân biệt MCAR/MAR."""
    return df.groupby(group_col, observed=True)[target_col].apply(lambda s: s.isna().mean())


def add_missing_indicator(df: pd.DataFrame, col: str, fill_value=0) -> pd.DataFrame:
    """Tạo cột {col}_was_missing rồi fillna. Dùng cho MCAR/MNAR khi cần giữ dấu vết."""
    df[f"{col}_was_missing"] = df[col].isna().astype(int)
    df[col] = df[col].fillna(fill_value)
    return df


def impute_by_group(df: pd.DataFrame, target_col: str, group_cols: list,
                     strategy: str = "median") -> pd.Series:
    """Impute theo nhóm — đúng cho MAR."""
    fn = {"median": lambda s: s.fillna(s.median()),
          "mean": lambda s: s.fillna(s.mean()),
          "mode": lambda s: s.fillna(s.mode().iloc[0])}[strategy]
    return df.groupby(group_cols, observed=True)[target_col].transform(fn)


# ============ W4 — Outlier & Consistency ============

def z_score_mask(s: pd.Series, threshold: float = 3) -> pd.Series:
    z = (s - s.mean()) / s.std()
    return z.abs() > threshold


def modified_z_mask(s: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Robust hơn z-score — dùng median + MAD."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=s.index)  # tránh chia 0
    m = 0.6745 * (s - med) / mad
    return m.abs() > threshold


def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return ~s.between(q1 - k * iqr, q3 + k * iqr)


def normalize_nfc(s: pd.Series) -> pd.Series:
    """Chuẩn hóa Unicode NFC — bắt buộc trước dedupe/group text tiếng Việt."""
    return s.str.strip().str.lower().apply(
        lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
    )


def run_checks(df: pd.DataFrame, checks: dict) -> None:
    """In số dòng vi phạm mỗi validity rule (checks: name -> boolean Series)."""
    for name, ok in checks.items():
        print(f"{name:30s}: {(~ok).sum():6d} rows violate it")


# ============ Cleaning log ============

def log_cleaning_step(log: list, step: str, column: str, finding: str,
                       action: str, n_rows: int, reason: str) -> None:
    """Append 1 dòng vào cleaning log (list of dict). Gọi ngay sau mỗi xử lý."""
    log.append({
        "step": step, "column": column, "finding": finding,
        "action": action, "n_rows": n_rows, "why": reason,
    })

# ============ Phase 2 — Reproducible raw-data contract ============
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_FILENAME = "ecommerce_customer_data_custom_ratios.csv"
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / RAW_FILENAME

EXPECTED_COLUMNS = (
    "Customer ID", "Purchase Date", "Product Category", "Product Price",
    "Quantity", "Total Purchase Amount", "Payment Method", "Customer Age",
    "Returns", "Customer Name", "Age", "Gender", "Churn",
)


def unexpected_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns outside the documented 13-column raw schema."""
    return [col for col in frame.columns if col not in EXPECTED_COLUMNS]


def validate_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """A non-mutating, per-column schema audit for the original Kaggle CSV."""
    rows = []
    for column in EXPECTED_COLUMNS:
        present = column in frame.columns
        rows.append({
            "column": column,
            "present": present,
            "dtype": str(frame[column].dtype) if present else "MISSING",
            "n_missing": int(frame[column].isna().sum()) if present else None,
        })
    for column in unexpected_columns(frame):
        rows.append({
            "column": column,
            "present": True,
            "dtype": str(frame[column].dtype),
            "n_missing": int(frame[column].isna().sum()),
        })
    return pd.DataFrame(rows)


def binary_domain_report(frame: pd.DataFrame, columns=("Returns", "Churn")) -> pd.DataFrame:
    """Report invalid observed binary values; missing is separate from invalid."""
    rows = []
    for column in columns:
        if column not in frame:
            rows.append({"column": column, "n_missing": None,
                         "n_invalid": None, "observed_values": "COLUMN_MISSING"})
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (~values.isin([0, 1]))
        rows.append({
            "column": column,
            "n_missing": int(frame[column].isna().sum()),
            "n_invalid": int(invalid.sum()),
            "observed_values": str(sorted(values.dropna().unique().tolist())),
        })
    return pd.DataFrame(rows)


def load_ecommerce_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read only the exact raw Kaggle file; never silently use processed data.

    The raw dataset is intentionally not committed. Missing raw data must fail
    with actionable instructions instead of fabricating a source from processed CSV.
    """
    source = Path(path) if path is not None else RAW_DATA_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Required raw dataset is missing: {source}\n"
            "Download the Kaggle dataset and place "
            f"{RAW_FILENAME} in data/raw/. "
            "See docs/phase2_runbook.md. No processed CSV is an acceptable substitute."
        )
    frame = pd.read_csv(source, low_memory=False)
    missing = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Raw schema is missing columns: {missing} (source: {source})")
    return frame
