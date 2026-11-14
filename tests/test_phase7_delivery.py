"""Phase 7 final delivery checks use only frozen aggregate evidence, no raw data."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from src.run_phase7 import (
    LEGACY_TRACKED, REQUIRED_DOCS, REQUIRED_EVIDENCE, ROOT,
    audit_repository, format_audit, main,
)


class Phase7AuditTests(unittest.TestCase):
    def test_repo_report_is_consistent(self):
        audit = audit_repository(ROOT)
        self.assertTrue(audit["technical_ready"], audit["checks"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["frozen_results"]["w11_confusion_matrix_tn_fp_fn_tp"], [3, 7886, 1, 1936])
        self.assertEqual(audit["frozen_results"]["customers"], 49673)

    def test_release_gate_never_automatically_granted(self):
        audit = audit_repository(ROOT)
        self.assertEqual(audit["release_clearance"], "NOT_CLEARED_FOR_PUBLIC_DATA_REDISTRIBUTION")
        self.assertEqual(audit["merge_status"], "DEFERRED_BY_USER")
        self.assertIn("data/processed/ecommerce_customer_cleaned.csv", audit["legacy_files_present"])
        self.assertEqual(set(audit["legacy_files_present"]), set(LEGACY_TRACKED))
        self.assertIn("NOT CLEARED", format_audit(audit))

    def test_missing_docs_and_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = audit_repository(root)
            self.assertFalse(report["technical_ready"])
            self.assertEqual(len(report["missing_documents"]), len(REQUIRED_DOCS))
            self.assertEqual(len(report["missing_evidence"]), len(REQUIRED_EVIDENCE))

    def _build_fixture(self, root: Path):
        for relative in (*REQUIRED_DOCS, *REQUIRED_EVIDENCE):
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
        for figure in (ROOT / "reports/figures").glob("*.png"):
            dest = root / "reports/figures" / figure.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure, dest)

    def test_tampered_classification_metrics_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            filename = root / "reports/phase5/classification_metrics.json"
            values = json.loads(filename.read_text(encoding="utf-8"))
            values["models"]["Random Forest"]["test"]["confusion_matrix_tn_fp_fn_tp"] = [0, 7889, 0, 1937]
            filename.write_text(json.dumps(values), encoding="utf-8")
            report = audit_repository(root)
            self.assertFalse(report["checks"]["w11_frozen_holdout"])
            self.assertFalse(report["technical_ready"])

    def test_tampered_fairness_holdout_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            filename = root / "reports/phase6/fairness_metrics.json"
            values = json.loads(filename.read_text(encoding="utf-8"))
            values["n_holdout_customers"] = 9999
            filename.write_text(json.dumps(values), encoding="utf-8")
            self.assertFalse(audit_repository(root)["checks"]["w12_frozen_fairness_holdout"])

    def test_tampered_phase51_decision_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            filename = root / "reports/phase5/phase51_metrics.json"
            values = json.loads(filename.read_text(encoding="utf-8"))
            values["model_replacement_approved"] = True
            filename.write_text(json.dumps(values), encoding="utf-8")
            self.assertFalse(audit_repository(root)["checks"]["phase51_no_false_upgrade"])

    def test_runner_writes_two_only_without_modifying_frozen_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            metrics = ROOT / "reports/phase5/classification_metrics.json"
            before = metrics.read_bytes()
            rc = main(["--output-dir", str(dest)])
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["release_audit.json", "release_audit.md"])
            self.assertEqual(before, metrics.read_bytes())
            doc = json.loads((dest / "release_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(doc["technical_ready"])
            self.assertEqual(doc["merge_status"], "DEFERRED_BY_USER")

    def test_dry_run_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            self.assertEqual(main(["--dry-run", "--output-dir", str(dest)]), 0)
            self.assertFalse(dest.exists())

    def test_all_new_internal_markdown_links_resolve(self):
        paths = ["README.md", "docs/reproducibility_guide.md", "docs/dataset_datasheet.md",
                 "docs/defense_guide.md", "docs/delivery_checklist.md",
                 "reports/final_project_report.md"]
        links = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for rel in paths:
            doc = ROOT / rel
            for match in links.finditer(doc.read_text(encoding="utf-8")):
                target = match.group(1).split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                with self.subTest(document=rel, link=target):
                    self.assertTrue((doc.parent / target).is_file(), f"Broken link {rel}: {target}")

    def test_no_person_level_artifacts_in_phase7_audit(self):
        audit = audit_repository(ROOT)
        combined = json.dumps(audit, ensure_ascii=False)
        # The audit intentionally contains only aggregate facts and policy warnings,
        # not a customer-level table or prediction list.
        self.assertNotIn("customer_rows", audit)
        self.assertNotIn("customer_predictions", combined)
        self.assertEqual(audit["frozen_results"]["w12_holdout_customers"], 9826)


if __name__ == "__main__":
    unittest.main()
