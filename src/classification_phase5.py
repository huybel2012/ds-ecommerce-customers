"""Phase 5 / W11: retrospective, customer-level synthetic Churn classification.

Churn has no documented event/label timestamp. A fixed feature observation cutoff
prevents post-cutoff features, but DOES NOT validate prospective churn prediction.
No person-level decisions should be made from this synthetic demonstration.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eda_metrics import AMOUNT, validate_eda_data

SEED = 42
CUTOFF = pd.Timestamp("2023-01-01")  # fixed before seeing holdout performance
CATEGORIES = ("Books", "Clothing", "Electronics", "Home")
PAYMENTS = ("Cash", "Credit Card", "Crypto", "PayPal")
FORBIDDEN_PREDICTORS = frozenset({
    "Customer ID", "Customer Name", "Churn", "Returns", "Returns_was_missing",
    "Gender", "Age", "Customer Age", "Purchase Date",
})


def make_customer_features(data: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """Deterministic pre-cutoff feature extraction; no learned global statistics.

    Labels are static customer metadata with UNKNOWN effective date. Their source
    is the entire dataset, but features strictly come from pre-cutoff events.
    This does NOT imply the labels were measured after the cutoff.
    """
    needed = {"Customer ID", "Purchase Date", "Product Category", "Payment Method",
              "Product Price", "Quantity", AMOUNT, "Churn", "Gender", "Age"}
    if missing := needed - set(data.columns):
        raise ValueError(f"Missing columns required for classification: {sorted(missing)}")
    if data.empty:
        raise ValueError("Empty dataset")
    df = data.copy()
    df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="raise")
    if df[list(needed)].isna().any().any():
        raise ValueError("Unexpected missing values in model input")
    if not df["Churn"].isin([0, 1]).all():
        raise ValueError("Churn must be binary")
    for field in ("Churn", "Gender", "Age"):
        if df.groupby("Customer ID", sort=False)[field].nunique(dropna=False).max() != 1:
            raise ValueError(f"Inconsistent customer-level {field}")
    if not df["Product Category"].isin(CATEGORIES).all() or not df["Payment Method"].isin(PAYMENTS).all():
        raise ValueError("Unexpected product/payment domain; update feature contract explicitly")
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff):
        raise ValueError("Invalid cutoff")
    history = df.loc[df["Purchase Date"] < cutoff].copy()
    if history.empty:
        raise ValueError("No transactions before observation cutoff")
    history["line_value"] = history["Product Price"].astype(float) * history["Quantity"].astype(float)
    g = history.groupby("Customer ID", sort=True, observed=True)
    agg = g.agg(
        transaction_count=("Purchase Date", "size"),
        first_purchase=("Purchase Date", "min"),
        last_purchase=("Purchase Date", "max"),
        active_days=("Purchase Date", lambda s: s.dt.normalize().nunique()),
        categories_count=("Product Category", "nunique"),
        payment_methods_count=("Payment Method", "nunique"),
        quantity_mean=("Quantity", "mean"),
        quantity_sum=("Quantity", "sum"),
        unit_price_mean=("Product Price", "mean"),
        recorded_amount_mean=(AMOUNT, "mean"),
        calculated_line_value_mean=("line_value", "mean"),
    )
    agg["recency_days"] = ((cutoff - agg["last_purchase"]).dt.total_seconds() / 86400).astype(float)
    agg["history_span_days"] = ((agg["last_purchase"] - agg["first_purchase"]).dt.total_seconds() / 86400).astype(float)
    features = agg.drop(columns=["first_purchase", "last_purchase"])
    for col, levels, prefix in (("Product Category", CATEGORIES, "category_share"),
                                ("Payment Method", PAYMENTS, "payment_share")):
        shares = pd.crosstab(history["Customer ID"], history[col], normalize="index")
        shares = shares.reindex(index=features.index, columns=list(levels), fill_value=0.0)
        shares.columns = [f"{prefix}_{v.lower().replace(' ', '_')}" for v in levels]
        features = features.join(shares)
    features = features.astype(float)
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Non-finite feature values")
    if set(features.columns) & FORBIDDEN_PREDICTORS:
        raise AssertionError("Forbidden feature included")
    metadata = df.drop_duplicates(subset="Customer ID").set_index("Customer ID").loc[features.index]
    labels = metadata["Churn"].astype(int).rename("Churn")
    subgroup = metadata[["Gender", "Age"]].copy()  # audit only, NEVER predictors
    audit = {
        "cutoff_exclusive": cutoff.isoformat(),
        "total_transactions": int(len(df)),
        "history_transactions": int(len(history)),
        "excluded_post_cutoff_transactions": int(len(df) - len(history)),
        "total_customers": int(df["Customer ID"].nunique()),
        "eligible_customers": int(len(features)),
        "ineligible_customers": int(df["Customer ID"].nunique() - len(features)),
        "positive_customers": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "latest_feature_transaction": history["Purchase Date"].max().isoformat(),
        "label_timestamp_known": False,
        "prospective_claim_permitted": False,
        "feature_columns": list(features.columns),
        "forbidden_predictors": sorted(FORBIDDEN_PREDICTORS),
        "source_limitation": "Static Churn label lacks a timestamp; retrospective synthetic classification only.",
    }
    assert history["Purchase Date"].max() < cutoff
    return features, labels, subgroup, audit


def customer_holdout(X: pd.DataFrame, y: pd.Series, test_size: float = .2, seed: int = SEED):
    """Group split even though the feature table has one row per customer."""
    if not X.index.is_unique or not y.index.equals(X.index):
        raise ValueError("One row per unique customer is required")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(X, y, groups=X.index.to_numpy()))
    if set(X.index[tr]) & set(X.index[te]):
        raise AssertionError("Customer leakage between train/test")
    return tr, te


def grouped_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 3, seed: int = SEED):
    if y.value_counts().min() < n_splits:
        raise ValueError("Too few positive/negative customers for grouped CV")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups=X.index.to_numpy()))
    for tr, va in folds:
        if set(X.index[tr]) & set(X.index[va]):
            raise AssertionError("Customer leakage inside CV")
        if y.iloc[tr].nunique() < 2 or y.iloc[va].nunique() < 2:
            raise ValueError("Both classes must occur in each training/validation fold")
    return folds


def choose_f1_threshold(y_true, prob, thresholds=None) -> tuple[float, float]:
    """Optimize train OOF F1 only; test labels are never passed here."""
    if thresholds is None:
        thresholds = np.round(np.arange(.05, .951, .025), 3)
    scores = np.array([f1_score(y_true, np.asarray(prob) >= t, zero_division=0) for t in thresholds])
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def classifier_models(trees: int = 80, seed: int = SEED) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=trees, max_depth=9, min_samples_leaf=25,
                class_weight="balanced_subsample", n_jobs=1, random_state=seed,
            )),
        ]),
    }


def score_predictions(y, probabilities, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities outside [0,1]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("ROC/PR require both classes")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC_average_precision": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
    }


def customer_bootstrap_ci(y, prob, threshold: float, n_boot: int = 250, seed: int = SEED) -> dict:
    """One holdout row per customer; bootstrap independent customers."""
    if n_boot < 20:
        raise ValueError("Need at least 20 bootstrap resamples")
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    scores = {name: [] for name in ("F1", "ROC_AUC", "PR_AUC_average_precision")}
    for _ in range(n_boot):
        index = rng.integers(0, len(y), size=len(y))
        yb, pb = y[index], prob[index]
        if len(np.unique(yb)) < 2:
            continue
        scores["F1"].append(f1_score(yb, pb >= threshold, zero_division=0))
        scores["ROC_AUC"].append(roc_auc_score(yb, pb))
        scores["PR_AUC_average_precision"].append(average_precision_score(yb, pb))
    if len(scores["F1"]) < n_boot // 2:
        raise ValueError("Too few valid bootstrap samples")
    return {metric: [float(q) for q in np.quantile(vals, [.025, .975])]
            for metric, vals in scores.items()}


def train_evaluate(X: pd.DataFrame, y: pd.Series, *, trees: int = 80,
                   folds: int = 3, bootstrap: int = 250, seed: int = SEED) -> tuple[dict, dict]:
    tr, te = customer_holdout(X, y, seed=seed)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]
    cv_splits = grouped_cv(Xtr, ytr, n_splits=folds, seed=seed)
    result = {
        "seed": seed, "train_customers": int(len(tr)), "test_customers": int(len(te)),
        "positive_rate_train": float(ytr.mean()), "positive_rate_test": float(yte.mean()),
        "split": "GroupShuffleSplit by Customer ID, one feature row/customer; 20% holdout",
        "cv": f"StratifiedGroupKFold({folds}) on training customers only",
        "model_selection": "Max training-only out-of-fold F1; no test set tuning",
        "threshold_optimization": "Training OOF F1 across thresholds 0.05-0.95 (step 0.025)",
        "models": {},
        "feature_columns": list(X.columns),
        "n_features": len(X.columns),
    }
    baseline = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    always_positive = DummyClassifier(strategy="constant", constant=1).fit(Xtr, ytr)
    scores = {"Prior baseline": baseline.predict_proba(Xte)[:, 1],
              "All-positive baseline": always_positive.predict_proba(Xte)[:, 1]}
    for base_name in ("Prior baseline", "All-positive baseline"):
        result["models"][base_name] = {
            "train_oof_F1": None, "threshold_source": "Fixed 0.5, no tuning",
            "test": score_predictions(yte, scores[base_name], .5),
        }
    result["train_all_positive_f1"] = float(f1_score(ytr, np.ones(len(ytr), dtype=int)))
    trained = {}
    for name, estimator in classifier_models(trees=trees, seed=seed).items():
        oof = cross_val_predict(estimator, Xtr, ytr, cv=cv_splits,
                                method="predict_proba", n_jobs=1)[:, 1]
        threshold, oof_f1 = choose_f1_threshold(ytr.to_numpy(), oof)
        fitted = estimator.fit(Xtr, ytr)
        prob = fitted.predict_proba(Xte)[:, 1]
        scores[name] = prob
        trained[name] = fitted
        result["models"][name] = {
            "train_oof_F1": oof_f1,
            "threshold_source": "Training-only grouped out-of-fold F1",
            "test": score_predictions(yte, prob, threshold),
            "test_at_0_5": score_predictions(yte, prob, .5),
        }
    chosen = max(trained, key=lambda nm: result["models"][nm]["train_oof_F1"])
    selected = result["models"][chosen]["test"]
    result["selected_by_train_only"] = chosen
    result["selected_train_oof_f1_gain_vs_all_positive"] = float(
        result["models"][chosen]["train_oof_F1"] - result["train_all_positive_f1"])
    result["selected_train_oof_beats_all_positive"] = bool(
        result["selected_train_oof_f1_gain_vs_all_positive"] > 0)
    result["selected_holdout_f1_gain_vs_all_positive"] = float(
        selected["F1"] - result["models"]["All-positive baseline"]["test"]["F1"])
    result["deployment_recommendation"] = "NOT FOR DEPLOYMENT: synthetic static Churn label has no event timestamp."
    result["selected_holdout_ci95"] = customer_bootstrap_ci(
        yte.to_numpy(), scores[chosen], selected["threshold"], n_boot=bootstrap, seed=seed)
    result["limitations"] = [
        "Synthetic Kaggle sample: not evidence of real-world performance.",
        "Static churn labels lack event/effective timestamp. Predicting future churn is NOT validated.",
        "Cutoff restricts feature history only; does not establish temporal validity of labels.",
        "Recorded amount/currency semantics remain unverified; no causal interpretation.",
        "No Returns, missingness flags, customer identifiers, demographic attributes, or names used as predictors.",
        "Model/threshold selected only using training OOF; holdout evaluated once.",
    ]
    # Detail is used only for visualization, never for the JSON report.
    detail = {"y_test": yte.to_numpy(), "probabilities": scores,
              "trained": trained, "chosen": chosen, "X_train": Xtr,
              "columns": list(X.columns)}
    return result, detail


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)


def make_report(audit: dict, result: dict) -> str:
    lines = [
        "# Phase 5 — W11: Retrospective Synthetic Churn Classification",
        "",
        "> **Không phải dự đoán churn tương lai:** Nhãn `Churn` cố định theo khách, không có timestamp xác nhận.",
        "> Dữ liệu mô phỏng; không dùng mô hình để quyết định về khách hàng thực.",
        "",
        "## Bài toán, thời điểm, đơn vị phân tích",
        f"- Snapshot: các giao dịch trước **{audit['cutoff_exclusive']}** (exclusive).",
        f"- {audit['history_transactions']:,} giao dịch lịch sử; **{audit['eligible_customers']:,} khách** đủ dữ liệu; loại {audit['ineligible_customers']:,} khách không có lịch sử trước cutoff.",
        f"- Positive (Churn=1): {audit['positive_customers']:,}/{audit['eligible_customers']:,} ({audit['positive_rate']*100:.2f}%).",
        "- Nhãn lấy từ thuộc tính khách trong dataset và **không có mốc xác lập**; đây là bài toán phân loại hồi cứu minh họa, không đủ bằng chứng cho prospectively valid churn forecasting.",
        "- Features chỉ từ giao dịch trước cutoff; không dùng Returns, Name, ID, Gender, Age, Churn làm predictors.",
        "- `recorded_amount_mean` là giá trị ghi nhận mô phỏng, **không phải doanh thu thực**; đơn vị tiền chưa rõ.",
        "",
        "## Chia tập, CV và lựa chọn threshold",
        f"- Group holdout theo Customer ID: train {result['train_customers']:,}; test {result['test_customers']:,} khách độc lập.",
        f"- {result['cv']}; threshold tối ưu bằng **OOF train F1**, không sử dụng test labels.",
        f"- Chọn mô hình dựa trên OOF train F1: **{result['selected_by_train_only']}**.",
        "- Baselines gồm `DummyClassifier(prior)` (đa số không Churn) và **All-positive** (đoán mọi khách đều Churn), để tránh hiểu F1 cao do đoán hết lớp dương là tín hiệu thật.",
        f"- F1 all-positive trên train: {result['train_all_positive_f1']:.4f}; chênh OOF train F1 so với baseline: **{result['selected_train_oof_f1_gain_vs_all_positive']:+.6f}** (dù dương rất nhỏ cũng không được coi là cải thiện hữu ích).",
        f"- Chênh F1 của model chọn so với all-positive trên **test**: **{result['selected_holdout_f1_gain_vs_all_positive']:+.6f}**. Mô hình vẫn chỉ là bài thực hành, KHÔNG triển khai thực tế.",
        "",
        "## Kết quả trên holdout",
        "| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | Balanced Acc | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        z = model["test"]
        lines.append("| " + name + " | " + " | ".join(_fmt(z[k]) for k in
            ("threshold", "precision", "recall", "F1", "ROC_AUC", "PR_AUC_average_precision", "balanced_accuracy", "accuracy")) + " |")
    lines.extend(["", "## Confusion matrix trên holdout", "", "Format `[TN, FP, FN, TP]`:"])
    for name, model in result["models"].items():
        lines.append(f"- **{name}:** {model['test']['confusion_matrix_tn_fp_fn_tp']}")
    lines.extend(["", "## Khoảng tin cậy 95% (bootstrap khách hàng)"])
    for metric, ci in result["selected_holdout_ci95"].items():
        lines.append(f"- {metric}: [{ci[0]:.4f}, {ci[1]:.4f}]")
    lines.extend([
        "", "## Feature engineering",
        f"- {result['n_features']} đặc trưng hành vi: " + ", ".join(result["feature_columns"]),
        "- Mỗi khách một dòng. Các tỷ lệ danh mục/phương thức được tính trên lịch sử của chính khách hàng đó.",
        "- Không fit imputer/scaler trên toàn dataset: các phép biến đổi nằm trong sklearn Pipeline và fit trên train/CV fold.",
        "", "## Hạn chế và khuyến nghị",
        "- Không diễn giải feature importance là quan hệ nhân quả.",
        "- Nếu ROC-AUC gần 0.5 và PR-AUC gần prevalence, kết luận tín hiệu phân loại yếu thay vì tô hồng Accuracy.",
        "- Cần event timestamp của Churn và dữ liệu as-of snapshots thật trước khi xây hệ thống dự đoán tương lai.",
        "- Phân tích subgroup/fairness chi tiết được thực hiện ở Phase 6.", "",
    ])
    return "\n".join(lines)


def save_figures(result: dict, detail: dict, figure_dir: Path) -> list[Path]:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    y = detail["y_test"]
    names = list(detail["probabilities"])
    colors = {"Prior baseline": "#64748b", "All-positive baseline": "#a3a3a3",
              "Logistic Regression": "#2563eb", "Random Forest": "#e8791b"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140)
    for name in ("Logistic Regression", "Random Forest"):
        prob = detail["probabilities"][name]
        fpr, tpr, _ = roc_curve(y, prob)
        precision, recall, _ = precision_recall_curve(y, prob)
        ax1.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC={roc_auc_score(y,prob):.3f})")
        ax2.plot(recall, precision, color=colors[name], label=f"{name} (AP={average_precision_score(y,prob):.3f})")
    # Constant-score classifiers have degenerate PR/ROC points; do not connect
    # their endpoints into a misleading diagonal "curve". The random ROC and
    # class prevalence PR reference represent their no-ranking performance.
    ax1.plot([0, 1], [0, 1], '--', color='#94a3b8', linewidth=1, label='Random reference')
    ax2.axhline(y.mean(), linestyle='--', color='#94a3b8', linewidth=1, label=f'Prevalence={y.mean():.3f}')
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — customer holdout", xlim=(0,1), ylim=(0,1))
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall — customer holdout", xlim=(0,1), ylim=(0,1))
    for ax in (ax1,ax2):
        ax.legend(fontsize=7,loc='best')
        ax.grid(alpha=.15)
    fig.suptitle("Synthetic Churn labels without event timestamp — NOT future prediction", fontsize=10)
    fig.tight_layout()
    p1=figure_dir/"10_classification_roc_pr.png"
    fig.savefig(p1,bbox_inches='tight');plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.5), dpi=140)
    for ax,name in zip(axs,names):
        tn,fp,fn,tp = result["models"][name]["test"]["confusion_matrix_tn_fp_fn_tp"]
        grid=np.array([[tn,fp],[fn,tp]])
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(1,grid.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{grid[i,j]:,}",ha='center',va='center',fontsize=10,
                        color='white' if grid[i,j]>grid.max()*.55 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title=name)
        ax.set_xticklabels(['0','1']);ax.set_yticklabels(['0','1'])
    fig.suptitle("Confusion matrices on untouched customer holdout", fontsize=11)
    fig.tight_layout()
    p2=figure_dir/"11_classification_confusion_matrices.png"
    fig.savefig(p2,bbox_inches='tight');plt.close(fig)

    chosen=detail["chosen"]
    model=detail["trained"][chosen].named_steps["model"]
    if chosen=="Logistic Regression":
        imp=np.abs(model.coef_[0]); label='|standardized logistic coefficient|'
    else:
        imp=model.feature_importances_;label='RF impurity importance (not causal)'
    top=np.argsort(imp)[-12:]
    fig,ax=plt.subplots(figsize=(8,4.5),dpi=140)
    ax.barh(np.array(detail["columns"])[top],imp[top],color='#2563eb')
    ax.set(xlabel=label,title=f"Training-only feature ranking · {chosen}")
    ax.set_xlim(left=0)
    fig.tight_layout()
    p3=figure_dir/"12_classification_feature_importance.png"
    fig.savefig(p3,bbox_inches='tight');plt.close(fig)
    return [p1,p2,p3]


def run_classification(df: pd.DataFrame, report_path: Path, metrics_path: Path,
                       figure_dir: Path, cutoff: pd.Timestamp = CUTOFF,
                       *, trees: int = 80, folds: int = 3, bootstrap: int = 250,
                       seed: int = SEED) -> dict:
    data=validate_eda_data(df)  # Phase 2 contract + Returns NULL preserved
    X,y,meta,audit=make_customer_features(data, cutoff=cutoff)
    result,detail=train_evaluate(X,y,trees=trees,folds=folds,bootstrap=bootstrap,seed=seed)
    result["feature_audit"]=audit
    result["subgroup_columns_available_for_phase6"]=["Gender", "Age"]
    reports=[Path(report_path),Path(metrics_path)]
    for p in reports:p.parent.mkdir(parents=True,exist_ok=True)
    # Run calculations/plots before publishing reports. No sensitive data serialized.
    figures=save_figures(result,detail,figure_dir)
    report_path.write_text(make_report(audit,result),encoding='utf-8')
    metrics_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return {"summary":result,"artifacts":reports+figures}
