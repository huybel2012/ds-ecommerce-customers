# Phase 6: Ethics, Fairness & Responsible AI (W12)

**Scope:** #09 ethics/privacy, #21 subgroup fairness & scalability, #10 model card & AI disclosure. Branch `Dung`, based on Phase 5.1 (`32ee9df`). Python 3.11.

## Commands (PowerShell)

```powershell
cd C:\Code\KHDL\Project
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.run_phase6 --dry-run
.\venv\Scripts\python.exe -m src.run_phase6
```

Full run should print: `PASS: Phase 6, frozen Phase 5 decisions verified, source unchanged, 3 artifacts.`

## Contract and output

The runner only reads `data/processed/ecommerce_customer_clean.csv` (Phase 2 clean data) and the committed `reports/phase5/classification_metrics.json` from Phase 5. It never falls back to legacy data. Files created:

- `reports/phase6/fairness_metrics.json`: group aggregate metrics and descriptive gaps; no per-person rows
- `reports/phase6/ethics_and_bias.md`: explain methods, synthetic source, limits, subgroup metrics and governance
- `reports/phase6/model_card.md`: one-page model card for W12
- `reports/phase6/run_manifest.json`: SHA-256 and machine-specific generation time; ignored by Git

The runner retrains Phase 5's **frozen Random Forest** with the same training customers, seed=42, 19 features and forest hyperparameters; it requires an exact match in the committed confusion matrix, precision, recall, F1, and balanced accuracy; small ranking-metric differences are allowed (ROC-AUC/AP absolute delta <=0.005) due to library-version differences, recorded explicitly as deltas rather than falsely claimed exact. It does not re-select the model, choose a new threshold, or inspect the test data for parameter tuning. Divergence fails closed. Frozen threshold was chosen using Phase 5 train-only OOF F1; it is NOT a sound deployment threshold.

## Fairness evaluation

- One person = one customer; demographics are taken from customer metadata, never input as predictors.
- Report Gender as dataset-defined Male/Female; age as six bands `18-25`, `26-35`, `36-45`, `46-55`, `56-65`, `66-70`; intersection Gender × Age.
- Per group: N, class support, positive selection rate, sensitivity (TPR), false positive rate (FPR), precision/PPV, balanced accuracy, ROC-AUC, average precision. Wilson intervals for rates are in JSON.
- For a group with N<50, positives<20, or negatives<20, suppress its performance metrics (small-cell protection); still report N and overall label counts from synthetic dataset.
- Descriptive group gaps = max minus min for supported groups, **not** a proof of discrimination, fairness, or statistical significance.
- A near-all-positive classifier can show near-equal TPR and selection rates *trivially*, while producing very large false-positive counts. All metrics must be interpreted with baseline, prevalence, and near-chance ROC-AUC.
- There is no opportunity to verify real intersectional representativeness: binary-only Gender and ages 18–70 are limitations of the source.

## Ethics & governance acceptance

Review `docs/ethics_checklist.md` and `docs/ai_usage_disclosure.md`. **Do not mark the phase legally cleared** until the dataset license is confirmed from its primary Kaggle metadata. The repository currently tracks legacy data with synthetic `Customer Name` and `Customer ID`; do not say it is fully anonymized. Do not upload customer-level predictions or raw extracts. Consent for actual customers is irrelevant to this synthetic exercise, but any real-world extension would require an independent privacy/legal review.

Before a public release, review whether the tracked legacy CSV and Parquet should be removed from Git history or replaced with data stripped of name and identifiers; `gitignore` alone cannot remove existing tracked files. That operation is intentionally **out of scope** for this patch and requires team approval.

## Performance and scale

250k transaction rows and ~50k customer feature rows are manageable with single-machine Pandas + NumPy + Parquet. Document peak RAM and runtime for each run before claiming production scalability. Use chunking/partitioned Parquet, pushdown, validation, storage and orchestration only at actual resource thresholds; Spark is not necessary to fulfill W12 for current data.

## Interpretation

- Phase 5 test model ROC-AUC≈0.495. Phase 5.1 repeated grouped CV is similarly near chance; no reliable signal found. The audit does **not fix accuracy or enable deployment**.
- The churn label has no event timestamp, and the synthetic nature of the data prevents real-world subgroup fairness validation.
- Follow the existing project roadmap: Phase 7 packages these reports and documents final human review, sources, attribution and licensing evidence.
