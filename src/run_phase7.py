"""Phase 7 / INFO3020: deterministic, read-only release and documentation audit.

Run ``python -m src.run_phase7 --dry-run`` to inspect, or run without flags to
write an aggregate audit. Does not require the raw Kaggle dataset. This audit is
NOT legal approval, ethics approval, or a substitute for a human final review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/phase7"

REQUIRED_DOCS = (
    "README.md",
    "docs/reproducibility_guide.md",
    "docs/dataset_datasheet.md",
    "docs/delivery_checklist.md",
    "docs/defense_guide.md",
    "docs/ai_usage_disclosure.md",
    "docs/ethics_checklist.md",
    "docs/scalability_assessment.md",
    "reports/final_project_report.md",
    "reports/phase6/model_card.md",
)
REQUIRED_EVIDENCE = (
    "reports/phase1/data_integrity_metrics.json",
    "reports/phase4/hypothesis_results.json",
    "reports/phase4/regression_metrics.json",
    "reports/phase5/classification_metrics.json",
    "reports/phase5/phase51_metrics.json",
    "reports/phase6/fairness_metrics.json",
)
LEGACY_TRACKED = (
    "data/processed/ecommerce_customer_cleaned.csv",
    "data/processed/ecommerce_clean.parquet",
)


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    result = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Invalid JSON object in {relative}")
    return result


def _approx(value: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(float(value) - expected) <= tol


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    """Check frozen repository results, never access/generate customer-level data."""
    root = Path(root)
    missing_docs = [p for p in REQUIRED_DOCS if not (root / p).is_file()]
    missing_evidence = [p for p in REQUIRED_EVIDENCE if not (root / p).is_file()]
    figures_missing = [f"reports/figures/{i:02d}_{name}.png" for i, name in {
        1: "category_revenue", 2: "purchase_amount_distribution", 3: "churn_rate_heatmap",
        4: "spending_by_age_gender", 5: "monthly_revenue_trend", 6: "residual_diagnostics",
        7: "coefficients_ci", 8: "actual_vs_predicted", 9: "hypothesis_tests",
        10: "classification_roc_pr", 11: "classification_confusion_matrices",
        12: "classification_feature_importance", 13: "phase51_univariate_signal",
        14: "phase51_repeated_cv",
    }.items() if not (root / f"reports/figures/{i:02d}_{name}.png").is_file()]
    checks: dict[str, bool] = {
        "required_documents_present": not missing_docs,
        "frozen_evidence_present": not missing_evidence,
        "all_14_figures_present": not figures_missing,
    }
    info: dict[str, Any] = {}
    if not missing_evidence:
        data = _read_json(root, REQUIRED_EVIDENCE[0])
        h = _read_json(root, REQUIRED_EVIDENCE[1])
        reg = _read_json(root, REQUIRED_EVIDENCE[2])
        cls = _read_json(root, REQUIRED_EVIDENCE[3])
        experiment = _read_json(root, REQUIRED_EVIDENCE[4])
        fairness = _read_json(root, REQUIRED_EVIDENCE[5])
        final = data.get("validated", {})
        # Dataset quality and observed gaps are frozen study facts; do not silently update.
        checks["integrity_data_counts"] = bool(
            final.get("rows") == 250000 and final.get("unique_customers") == 49673
            and final.get("returns_missing_count") == 47596
            and final.get("amount_mismatch_count") == 249952
        )
        checks["w9_holm_family_result"] = bool(
            all(h[key]["holm_p_fwer_3_tests"] >= 0.05 for key in ("H1", "H2", "H3"))
            and h.get("customers") == 49673
        )
        checks["w10_regression_result"] = bool(
            _approx(reg["models"]["OLS"]["R2"], .003222, .00015)
            and _approx(reg["ols_mae_gain_over_baseline"], 1.292, .02)
        )
        winner = cls.get("selected_by_train_only")
        winner_data = cls.get("models", {}).get(winner, {}).get("test", {})
        cm = winner_data.get("confusion_matrix_tn_fp_fn_tp", [])
        checks["w11_frozen_holdout"] = bool(
            winner == "Random Forest" and cls.get("test_customers") == 9826
            and cm == [3, 7886, 1, 1936]
            and _approx(winner_data.get("ROC_AUC", 0), .4948531813, .00001)
            and _approx(winner_data.get("threshold", -1), .375)
        )
        checks["phase51_no_false_upgrade"] = bool(
            experiment.get("passed_exploratory_screen") is False
            and experiment.get("model_replacement_approved") is False
            and experiment.get("held_out_customers_not_scored") == 9826
        )
        checks["w12_frozen_fairness_holdout"] = bool(
            fairness.get("n_holdout_customers") == cls.get("test_customers")
            and _approx(fairness.get("threshold", -1), winner_data.get("threshold", -2))
            and fairness.get("overall", {}).get("confusion_matrix_tn_fp_fn_tp") == cm
        )
        info = {
            "transactions": final.get("rows"),
            "customers": final.get("unique_customers"),
            "returns_missing": final.get("returns_missing_count"),
            "amount_mismatches": final.get("amount_mismatch_count"),
            "w9_holm_p": {k: round(h[k]["holm_p_fwer_3_tests"], 6) for k in ("H1", "H2", "H3")},
            "w10_ols_test_R2": reg["models"]["OLS"]["R2"],
            "w10_ols_mae_gain": reg["ols_mae_gain_over_baseline"],
            "w11_model": winner,
            "w11_threshold": winner_data.get("threshold"),
            "w11_roc_auc": winner_data.get("ROC_AUC"),
            "w11_confusion_matrix_tn_fp_fn_tp": cm,
            "w12_holdout_customers": fairness.get("n_holdout_customers"),
        }
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    checks["ai_disclosure_in_readme"] = (
        "AI assistance" in readme and "docs/ai_usage_disclosure.md" in readme
    )
    checks["no_main_merge_in_delivery_instructions"] = bool(
        (root / "docs/delivery_checklist.md").is_file()
        and "merge deferred" in (root / "docs/delivery_checklist.md").read_text(encoding="utf-8")
    )
    # Presence is a WARNING, not an instruction to delete history automatically.
    legacy_present = [p for p in LEGACY_TRACKED if (root / p).is_file()]
    technical_ready = all(checks.values())
    open_gates = [
        "Dataset Kaggle license/redistribution rights: verify from primary source, record evidence.",
        "Legacy tracked CSV/Parquet contain synthetic Customer Name/ID: approve publication or history remediation.",
        "No measured k-anonymity claim; no real-data consent/privacy clearance.",
        "Team identity, contribution evidence, AI-use statement and one-page model card need human sign-off.",
        "Final Windows run on canonical clean data and CI run at Phase 7 commit require confirmation.",
    ]
    return {
        "scope": "INFO3020 W1-W12 (W8 and exercise-only tasks excluded), Phase 7 without merge",
        "base_reference_commit": "2ab6bae",
        "technical_ready": technical_ready,
        "release_clearance": "NOT_CLEARED_FOR_PUBLIC_DATA_REDISTRIBUTION",
        "merge_status": "DEFERRED_BY_USER",
        "checks": checks,
        "missing_documents": missing_docs,
        "missing_evidence": missing_evidence,
        "missing_figures": figures_missing,
        "legacy_files_present": legacy_present,
        "frozen_results": info,
        "human_review_gates": open_gates,
    }


def format_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# Phase 7 — Repository Delivery Audit (W1–W12)", "",
        "**Scope:** Documentation, reproducibility, defense and release checks; **no merge or PR**.",
        f"**Technical readiness:** {'PASS' if audit['technical_ready'] else 'FAIL'} (automated checks only). ",
        "**Public redistribution:** NOT CLEARED. Do not infer data license, consent or anonymization. ",
        "**Merge:** deferred by request.", "", "## Automated checks", "",
        "| Check | Result |", "|---|---|",
    ]
    lines += [f"| `{key}` | {'PASS' if value else 'FAIL'} |" for key,value in audit["checks"].items()]
    lines += ["", "## Frozen result cross-check", "", "```json",
              json.dumps(audit["frozen_results"], ensure_ascii=False, indent=2), "```", "",
              "## Tracked legacy datasets (manual disposition)", ""]
    lines += [f"- `{p}`" for p in audit["legacy_files_present"]] or ["- None detected (does not establish historical removal)."]
    lines += ["", "## Gates needing human approval", ""]
    lines += [f"- [ ] {item}" for item in audit["human_review_gates"]]
    for key in ("missing_documents", "missing_evidence", "missing_figures"):
        if audit[key]:
            lines += ["", f"Missing {key}: " + ", ".join(f"`{x}`" for x in audit[key])]
    lines += ["", "This audit is evidence management, not an independent rerun of the 250k-row pipeline.", ""]
    return "\n".join(x.rstrip() for x in lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Audit without writing files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="Destination for audit report and JSON")
    args = parser.parse_args(argv)
    report = audit_repository(ROOT)
    print("Phase 7: technical checks", sum(report["checks"].values()), "/", len(report["checks"]))
    print("Technical readiness:", "PASS" if report["technical_ready"] else "FAIL")
    print("Public redistribution: NOT CLEARED (human review required)")
    print("Merge/PR: deferred by request")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "release_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "release_audit.md").write_text(format_audit(report), encoding="utf-8")
        print("Created:", args.output_dir / "release_audit.json")
        print("Created:", args.output_dir / "release_audit.md")
    return 0 if report["technical_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
