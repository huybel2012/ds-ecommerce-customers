# Phase 5: Classification & Feature Engineering (W11)

**Yêu cầu:** Nhánh `Dung` sau Phase 4; Python 3.11; dataset *raw* đã được tải từ Kaggle và đã qua Phase 2.

## Source

Dùng duy nhất `data/processed/ecommerce_customer_clean.csv` sinh từ `python -m src.run_pipeline` (có `Returns=NULL` và cờ missing). Chạy `--validate-only` nếu chưa chắc dữ liệu hiện tại đúng. Code không fallback sang CSV legacy.

## Lệnh Windows PowerShell

```powershell
cd C:\Code\KHDL\Project
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.run_phase5 --dry-run
.\venv\Scripts\python.exe -m src.run_phase5
```

## Outputs

- `reports/classification_results.md`: W11 narrative with holdout metrics and severe temporal labeling caveat.
- `reports/phase5/classification_metrics.json`: split, features, CV F1, chosen threshold and holdout metrics/CI.
- `reports/figures/10_classification_roc_pr.png`: ROC/PR curves and baseline prevalence.
- `reports/figures/11_classification_confusion_matrices.png`: held-out TN/FP/FN/TP.
- `reports/figures/12_classification_feature_importance.png`: ranking learned on train only, NOT causal.
- `reports/phase5/run_manifest.json`: local SHA256, timestamp and input provenance; **ignored by Git**.

Artifacts from Phase 1–4 are not overwritten. Dataset source is never modified.

## Training protocol

- Fixed cutoff **2023-01-01** (exclusive), one customer → one feature row.
- Numeric transaction-history features and category/payment shares only; exclude Returns/Churn/ID/demographics/names.
- One GroupShuffleSplit 80/20, customer disjoint; train-only 3-fold StratifiedGroupKFold.
- `DummyClassifier(prior)` and all-positive baselines, LogisticRegression and RandomForestClassifier; train OOF F1 chooses threshold and model without reading test labels.
- Holdout reporting: Accuracy, balanced accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC (average precision), Brier, confusion matrix; bootstrap (customers) 95% CI for selected model.
- **Do not interpret high Accuracy as strong classification if recall/F1/PR-AUC are weak.**

## Reproducibility acceptance

- All tests pass and dry-run identifies correct clean file.
- Full run prints `PASS: Phase 5, source unchanged, 5 artifacts` and creates 3 PNGs, 1 Markdown, 1 JSON + local manifest.
- Feature cutoff respected; no train/test ID overlap, no return imputation, metrics generated from unchanged source.
- End-to-end result must be checked on the **actual cleaned CSV from Kaggle** on Windows/Python 3.11 before commit. The repository ZIP contains a legacy CSV that is not a substitute.
- Graphs 01–09 remain untouched.

See `docs/churn_label_assessment.md` for label timestamp limitations. Phase 6 will assess subgroup fairness and ethics.
