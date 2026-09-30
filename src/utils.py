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