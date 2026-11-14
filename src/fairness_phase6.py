"""Phase 6 / W12: post-hoc, customer-level fairness audit of the frozen Phase 5 model.

Sensitive metadata is used ONLY for evaluation. This module does not tune a model,
change the threshold, save customer IDs, or claim real-world fairness from Faker data.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

from src.classification_phase5 import (
    CUTOFF, SEED, classifier_models, customer_holdout, make_customer_features,
    score_predictions,
)
from src.eda_metrics import validate_eda_data

AGE_BINS = [17, 25, 35, 45, 55, 65, 70]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-70"]
MIN_GROUP = 50
MIN_CLASS = 20
# These are descriptive monitoring signals, NOT legal/ethical pass/fail thresholds.
REPORT_METRICS = ("selection_rate", "tpr", "fpr", "ppv", "balanced_accuracy", "roc_auc", "average_precision")


def wilson_ci(success: int, count: int, confidence: float = .95) -> list[float] | None:
    if count <= 0:
        return None
    z = float(norm.ppf((1 + confidence) / 2))
    p = success / count
    denom = 1 + z*z/count
    center = (p + z*z/(2*count)) / denom
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denom
    return [max(0.0, center-half), min(1.0, center+half)]


def subgroup_frame(y: pd.Series, probabilities: np.ndarray, sensitive: pd.DataFrame,
                   threshold: float) -> pd.DataFrame:
    """Return one row per held-out customer; do not expose identifiers in outputs."""
    if not isinstance(y, pd.Series) or not y.index.is_unique:
        raise ValueError("Customer labels must be a unique-index Series")
    if not sensitive.index.is_unique or not sensitive.index.equals(y.index):
        raise ValueError("Subgroup metadata must align exactly with held-out customers")
    if len(probabilities) != len(y):
        raise ValueError("Probability/label length mismatch")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be frozen within (0,1)")
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability values")
    if not set(pd.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if sensitive[["Gender", "Age"]].isna().any().any():
        raise ValueError("Missing subgroup labels")
    if not sensitive["Gender"].isin(["Male", "Female"]).all():
        raise ValueError("Unexpected Gender domain: report must not silently drop people")
    age = pd.to_numeric(sensitive["Age"], errors="raise")
    if not age.between(18, 70).all():
        raise ValueError("Age outside dataset declared domain")
    age_band = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    if age_band.isna().any():
        raise ValueError("Age band assignment failed")
    # No Customer ID or Customer Name columns are ever materialized for reporting.
    return pd.DataFrame({
        "actual": y.to_numpy(dtype=int),
        "probability": p,
        "predicted": (p >= threshold).astype(int),
        "Gender": sensitive["Gender"].to_numpy(),
        "AgeBand": age_band.astype(str).to_numpy(),
    })


def _safe_ratio(a: int, b: int):
    return float(a / b) if b else None


def group_metrics(data: pd.DataFrame, key: str, *, min_group: int = MIN_GROUP,
                  min_class: int = MIN_CLASS) -> list[dict[str, Any]]:
    if min_group < 2 or min_class < 2:
        raise ValueError("min_group and min_class must be >=2")
    if key not in data.columns:
        raise ValueError(f"Unknown grouping dimension: {key}")
    rows = []
    for name, g in data.groupby(key, sort=True, dropna=False):
        n = len(g)
        y = g["actual"].to_numpy(dtype=int)
        pred = g["predicted"].to_numpy(dtype=int)
        p = g["probability"].to_numpy(dtype=float)
        pos, neg = int(y.sum()), int((1-y).sum())
        tp, fn = int(((y==1)&(pred==1)).sum()), int(((y==1)&(pred==0)).sum())
        fp, tn = int(((y==0)&(pred==1)).sum()), int(((y==0)&(pred==0)).sum())
        sel = tp + fp
        eligible = n >= min_group and min(pos, neg) >= min_class
        base = {"group": str(name), "n": n, "positives": pos, "negatives": neg,
                "suppressed": not eligible,
                "reason": None if eligible else "insufficient support for reliable subgroup reporting"}
        if not eligible:
            # Do not output disclosive cell-level confusion matrix for small groups.
            base.update({k: None for k in REPORT_METRICS})
            base.update({"confusion_matrix_tn_fp_fn_tp": None,
                         "selection_rate_ci95": None,"tpr_ci95": None,"fpr_ci95": None,
                         "ppv_ci95": None})
        else:
            base.update({
                "selection_rate": _safe_ratio(sel, n),
                "tpr": _safe_ratio(tp, pos),
                "fpr": _safe_ratio(fp, neg),
                "ppv": _safe_ratio(tp, sel),
                "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
                "roc_auc": float(roc_auc_score(y,p)),
                "average_precision": float(average_precision_score(y,p)),
                "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
                "selection_rate_ci95": wilson_ci(sel, n),
                "tpr_ci95": wilson_ci(tp, pos),
                "fpr_ci95": wilson_ci(fp, neg),
                "ppv_ci95": wilson_ci(tp, sel),
            })
        rows.append(base)
    return rows


def disparity(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r["suppressed"]]
    gaps = {}
    for metric in REPORT_METRICS:
        values = [r[metric] for r in valid if r[metric] is not None]
        gaps[metric+"_gap"] = (float(max(values)-min(values)) if len(values) >= 2 else None)
    t, f = gaps.get("tpr_gap"), gaps.get("fpr_gap")
    gaps["equalized_odds_gap"] = max(t, f) if t is not None and f is not None else None
    return gaps


def subgroup_audit(y: pd.Series, probabilities: np.ndarray, sensitive: pd.DataFrame,
                  threshold: float, *, min_group: int = MIN_GROUP,
                  min_class: int = MIN_CLASS) -> dict:
    df = subgroup_frame(y, probabilities, sensitive, threshold)
    df["GenderAge"] = df["Gender"] + "/" + df["AgeBand"]
    groups = {k: group_metrics(df, k, min_group=min_group, min_class=min_class)
              for k in ("Gender", "AgeBand", "GenderAge")}
    gaps = {k: disparity(v) for k,v in groups.items()}
    overall = score_predictions(df["actual"].to_numpy(), df["probability"].to_numpy(), threshold)
    return {"n_holdout_customers": int(len(df)), "threshold": float(threshold),
            "overall": overall,"groups": groups, "disparities": gaps,
            "support_rule": {"min_group_n":min_group,"min_positive_n":min_class,"min_negative_n":min_class},
            "limitations": [
                "Faker synthetic customer labels are not representative of protected real populations.",
                "Gender only has Male/Female; Age range 18-70. These are dataset categories, not an exhaustive identity taxonomy.",
                "The label lacks an effective timestamp; not prospective churn prediction.",
                "FPR/TPR or demographic parity near equal can be VACUOUS when almost every customer is predicted churn.",
                "Differences are descriptive; no hypothesis tests or legal fairness thresholds are claimed.",
                "Post-hoc fairness evaluation does not fix low model discrimination (ROC-AUC approximately 0.5).",
            ]}


def frozen_phase5_holdout(data: pd.DataFrame, phase5: dict, *, trees: int = 80) -> tuple[pd.Series, np.ndarray, pd.DataFrame, dict]:
    """Rebuild exact Phase 5 chosen model on its train subset without reselecting it.

    Verify computed metrics against committed Phase 5 outputs; fail closed on drift.
    """
    clean = validate_eda_data(data)
    X, y, subgroup, audit = make_customer_features(clean, cutoff=CUTOFF)
    audit['amount_mismatches'] = int((clean['Total Purchase Amount'] !=
                                    clean['Product Price'] * clean['Quantity']).sum())
    if phase5.get("selected_by_train_only") != "Random Forest":
        raise ValueError("Phase 5 chosen model drifted; review audit method explicitly")
    if phase5.get("seed") != SEED or phase5.get("feature_columns") != list(X.columns):
        raise ValueError("Phase 5 seed/feature contract differs; cannot compare")
    expected_audit = phase5.get("feature_audit", {})
    for field in ("eligible_customers", "history_transactions", "cutoff_exclusive"):
        if expected_audit.get(field) != audit.get(field):
            raise ValueError(f"Phase 5 source feature audit mismatch in {field}")
    train, test = customer_holdout(X, y, seed=SEED)
    if (len(train), len(test)) != (phase5.get("train_customers"),phase5.get("test_customers")):
        raise ValueError("Phase 5 holdout sizes changed")
    frozen = phase5["models"]["Random Forest"]["test"]
    threshold = float(frozen["threshold"])
    clf = classifier_models(trees=trees, seed=SEED)["Random Forest"]
    clf.fit(X.iloc[train],y.iloc[train])
    probs=clf.predict_proba(X.iloc[test])[:,1]
    observed=score_predictions(y.iloc[test].to_numpy(), probs, threshold)
    # The *binary decisions* must match exactly. Ranking metrics can differ
    # slightly between supported sklearn versions (tie handling/forest scores).
    # Record that discrepancy rather than silently claiming exact reproduction.
    required_exact = ("F1", "balanced_accuracy", "precision", "recall")
    for key in required_exact:
        if not np.isclose(observed[key], frozen[key], atol=1e-9, rtol=0):
            raise ValueError(f"Phase 5 frozen model metric drifted: {key}")
    if observed["confusion_matrix_tn_fp_fn_tp"] != frozen["confusion_matrix_tn_fp_fn_tp"]:
        raise ValueError("Phase 5 frozen model predictions drifted")
    ranking_tolerance = 0.005  # absolute; declared in report, not treated as exact
    for key in ("ROC_AUC", "PR_AUC_average_precision"):
        if abs(observed[key] - frozen[key]) > ranking_tolerance:
            raise ValueError(f"Phase 5 frozen model ranking metric drifted: {key}")
    audit["model_reproduction"] = {
        "classification_decisions_match": True,
        "ranking_metric_abs_tolerance": ranking_tolerance,
        "ranking_metrics_exact_match": all(abs(observed[k]-frozen[k])<1e-9 for k in ("ROC_AUC", "PR_AUC_average_precision")),
        "differences_current_minus_phase5": {k:float(observed[k]-frozen[k]) for k in ("ROC_AUC", "PR_AUC_average_precision", "F1")},
        "phase5_original_roc_auc":float(frozen['ROC_AUC']),
        "recomputed_roc_auc":float(observed['ROC_AUC']),
    }
    return y.iloc[test], probs, subgroup.iloc[test], audit


def make_fairness_report(audit: dict, res: dict, source_info: dict) -> str:
    m = res["overall"]
    lines = [
        "# Phase 6 — Ethics, Fairness & Responsible AI (W12)", "",
        "## Scope & method", "",
        "- Synthetic Faker Kaggle dataset. Not real-world customer fairness evidence.",
        "- Phase 5 **frozen Random Forest** and its training-only-selected threshold; exact customer holdout reproduced.",
        "- Sensitive `Gender`/`Age` used ONLY for post-hoc audit, not model training; one customer = one observation.",
        "- Raw/reconstructed source is validated and hashed. No customer identifiers/names/probabilities are exported.",
        "- Frozen classification decisions must match Phase 5 exactly; ranking AUC/AP allowed a documented absolute tolerance of 0.005 for library differences, with every discrepancy logged (NOT claimed as exact).",
        "- Gender: Male/Female only; age bins: 18-25 / 26-35 / 36-45 / 46-55 / 56-65 / 66-70.",
        "- Rates: TPR=TP/(TP+FN); FPR=FP/(FP+TN); PPV=TP/(TP+FP); selection_rate=(TP+FP)/n.",
        "- Wilson 95% intervals for binomial rates. Suppress cells with insufficient group/class support.",
        "", "## Overall holdout results", "",
        f"- N={res['n_holdout_customers']:,}; fixed threshold={res['threshold']:.3f}; ROC-AUC={m['ROC_AUC']:.4f}, PR-AUC={m['PR_AUC_average_precision']:.4f}, balanced accuracy={m['balanced_accuracy']:.4f}.",
        f"- TN/FP/FN/TP={m['confusion_matrix_tn_fp_fn_tp']}. Predictive validity is extremely limited.",
        f"- **Phase 5 replay:** ranking metric deltas: {audit['model_reproduction']['differences_current_minus_phase5']} (current minus committed Phase 5).",
        "", "## Subgroup performance", "",
    ]
    for key, label in (("Gender","Gender (dataset categories)"),("AgeBand","Age bins"),("GenderAge","Intersectional Gender × Age")):
        lines += [f"### {label}", "",
                  "| Group | N | Churn=1 | Predicted-positive rate | TPR | FPR | PPV | ROC-AUC |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in res["groups"][key]:
            def fmt(v):return "suppressed" if v is None else f"{v:.3f}"
            lines.append(f"| {row['group']} | {row['n']} | {row['positives']} | {fmt(row['selection_rate'])} | {fmt(row['tpr'])} | {fmt(row['fpr'])} | {fmt(row['ppv'])} | {fmt(row['roc_auc'])} |")
        g=res['disparities'][key]
        lines.append("")
        lines.append("**Descriptive gaps:** " + ", ".join(f"{k}={v:.3f}" for k,v in g.items() if v is not None and k in ("selection_rate_gap", "tpr_gap", "fpr_gap", "ppv_gap", "equalized_odds_gap")))
        lines.append("")
    lines += [
        "## Interpretation & limitations", "",
        "- **No fairness certification.** Similar selection rates can be trivial because nearly everyone is predicted positive, not evidence of equitable real-world impact.",
        "- Differences in TPR/FPR/PPV are sample descriptions only; CI widths and group prevalence need to be read alongside point estimates.",
        "- The label has no effective date, ROC-AUC is near chance, and training data is synthetic. Deployment is prohibited by this project protocol.",
        "- Small-group suppression prevents sparse-cell overinterpretation and avoids releasing tiny subgroup confusion matrices.",
        "- Gender categories exclude non-binary and unknown identities; age limited to 18–70. External validity not established.",
        "- Tracked legacy CSV/Parquet include synthetic `Customer Name`; never claim full de-identification or release new raw extracts.",
        "", "## Governance & licensing", "",
        "- Source: " + source_info["dataset_url"],
        "- Dataset reuse license: **NOT independently verified from the primary Kaggle license field**. Do not assert CC0/MIT or unrestricted redistribution.",
        "- Kaggle downloader should check current license, attribution and derivative redistribution terms before public distribution.",
        "- Do not use synthetic results to target, penalize, price-discriminate, or exclude actual people.",
        "- Ethical review is ongoing if legal permission, source semantics or stakeholder approval remains unverified.",
        "", "## Data operations & scalability", "",
        "- The data has 250k rows (~50k customers): Pandas/Parquet and a local Python 3.11 pipeline are sufficient for the demonstrated study.",
        "- No need to add Spark solely for W12; at larger scale prefer chunked/partitioned Parquet, key validation, memory profiling, orchestration and distributed joins only when measured memory/latency requirements demand them.",
        "", "## AI usage & human accountability", "",
        "- See `docs/ai_usage_disclosure.md`: AI-assisted code/report drafting and tests must be reviewed by student authors; no automated assistant guarantees correctness, legality, or ethics.",
        "", "## Decision", "",
        "**DO NOT DEPLOY.** Phase 6 measures limitations and subgroup discrepancies; it does not repair the weak predictive signal in Phase 5–5.1.", "",
    ]
    return "\n".join(lines)


def make_model_card(audit: dict, res: dict) -> str:
    m=res['overall'];g=res['disparities'];
    s=lambda v: "N/A" if v is None else f"{v:.3f}"
    return f"""# Model Card (1-page) — Retrospective Synthetic Churn Classifier

**Version / status:** Phase 6 · research demonstration ONLY · **NOT FOR DEPLOYMENT**
**Model:** Random Forest (Phase 5 frozen hyperparameters; {audit['eligible_customers']:,} eligible synthetic customers).
**Owner / accountable reviewers:** university project team; final human review required.
**Purpose:** illustrate customer-level binary classification and post-hoc subgroup evaluation for W11–W12; NOT forecast future churn, profile real people, set prices, or allocate interventions.

## Data & protocol
- **Source:** Shriyash Jagtap, [Kaggle e-commerce customer dataset](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis); synthetic Faker data ({audit['total_transactions']:,} transaction rows; {audit['total_customers']:,} customers). License **not verified from primary Kaggle metadata**.
- **Observation:** features from transactions before 2023-01-01, static `Churn` label has NO verified event/effective timestamp.
- **Inputs:** 19 aggregate behavior features; `Customer ID` used only to split, names/age/gender/Returns/label excluded from predictors. Age and Gender used for evaluation only.
- **Evaluation:** group-disjoint holdout N={res['n_holdout_customers']:,}, seed=42; training-only OOF F1 chose threshold={res['threshold']:.3f}; Phase 6 **replays**, not retunes, this decision. Binary confusion matrix must match exactly; any minor AUC/AP differences (<0.005 absolute) are recorded in fairness report.

## Frozen holdout results
- ROC-AUC={m['ROC_AUC']:.4f}; PR-AUC={m['PR_AUC_average_precision']:.4f}; F1={m['F1']:.4f}; balanced accuracy={m['balanced_accuracy']:.4f}.
- Confusion matrix [TN, FP, FN, TP]={m['confusion_matrix_tn_fp_fn_tp']}. Nearly all customers are predicted Churn; high recall here is NOT evidence of useful discrimination.
- Phase 5.1 added temporal features and ran shuffled-label controls; **no reliable improvement** was found.

## Fairness / uncertainty
- Post-hoc Gender, AgeBand and Gender×Age subgroup metrics include selection rate, TPR, FPR, PPV, ROC-AUC and Wilson rate intervals. See `reports/phase6/fairness_metrics.json`.
- Gender selection-rate gap={s(g['Gender']['selection_rate_gap'])}; AgeBand selection-rate gap={s(g['AgeBand']['selection_rate_gap'])}. Similar positive rates are **vacuous** for an all-positive-like classifier.
- Binary gender categories and restricted ages exclude many real identities; synthetic data cannot demonstrate equitable impact on real people. Small groups are suppressed; metrics are descriptive, not a fairness certification.

## Data quality / ethics / limitations
- `Returns` missingness was preserved by Phase 1–2; `Total Purchase Amount` differs from Price×Quantity for {audit['amount_mismatches']:,}/{audit['total_transactions']:,} rows; currency semantics unverified.
- The repository contains tracked synthetic names in legacy CSV/Parquet; no individual predictions or names are exported by this audit. Review reuse license before redistributing source data.
- No real-world validity, calibration, operational safety, or prospective forecast has been established. **Do not use for decisions affecting people.**

**AI-assistance:** AI tools contributed to draft code/tests/documentation; human verification and responsibility remain with student authors. See `docs/ai_usage_disclosure.md`.
"""


def generate_phase6(data: pd.DataFrame, phase5: dict, *, trees: int = 80,
                    min_group: int = MIN_GROUP, min_class: int = MIN_CLASS) -> tuple[dict,str,str]:
    y, proba, sensitive, audit = frozen_phase5_holdout(data,phase5,trees=trees)
    results=subgroup_audit(y,proba,sensitive,float(phase5['models']['Random Forest']['test']['threshold']),
                           min_group=min_group,min_class=min_class)
    results['provenance'] = {
        'phase5_metric_reproduction': audit['model_reproduction'],
        'phase5_model': 'Random Forest', 'phase5_snapshot_cutoff_exclusive': audit['cutoff_exclusive'],
        'frozen_training_seed': SEED, 'features_count': len(phase5['feature_columns']),
        'model_retrained_for_audit': True, 'model_selection_or_threshold_tuning_in_phase6': False,
        'synthetic_data': True, 'label_timestamp_known': False,
    }
    source={'dataset_url':'https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis'}
    return results,make_fairness_report(audit,results,source),make_model_card(audit,results)
