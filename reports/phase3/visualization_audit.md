# Phase 3 — EDA & Dashboard Visualization Audit

**Scope:** #12 (wrong category headline), #13 (five W7 charts), #14 (Streamlit KPI and sourcing).

**Dataset examined:** 250,000 synthetic records, 49,673 distinct customers, interval 2020-01-01 to 2023-09-15. Recorded amount values were checked using the ZIP's tracked legacy CSV for comparative chart QA. The **source of truth for the updated code** is the **Phase 2 cleaned CSV** (`data/processed/ecommerce_customer_clean.csv`); final sign-off must be confirmed on the user's machine. The old CSV is never a fallback in the updated code.

| Chart | Previous defect | Corrected metric / interpretation |
|---|---|---|
| 01. Category totals | Hardcoded Electronics + Clothing "lead" despite actual sums; called them USD revenue | Books = **204,939,601**, Clothing = **204,532,405**, Electronics = 136,599,467, Home = 135,271,210, using **sum(Total Purchase Amount)**; dynamic ordering; no asserted currency |
| 02. Amount distribution | Asserts uniform distribution from visually flat histogram and claims a hardcoded $500–$5,000 interval | Shows the empirical histogram, data median and actual range, without claiming a known distribution or currency |
| 03. Churn heatmap | Aggregated transaction rows, over-weighting customers with repeated purchases; fixed 20% uniformity narrative | Deduplicate `Customer ID` **within each Category × Payment Method cell**; explicitly say customers may appear in multiple cells and these are descriptive associations only |
| 04. Age/gender medians | Hardcoded claims of balance and USD units | Recomputed medians on available records, zero-based bar axis, unverified currency label, no causal assertions |
| 05. Monthly amounts | Hardcoded 12/2020 peak, $15M stability claims, final half-month drawn as if a full-month decline | Dynamically compute peak among complete boundary months, **disconnected orange marker** for 2023-09 (only to 15th), no USD or stable trend claim |

## KPI audit

- `reported_amount_total`: **681,342,683** in the source's unspecified monetary units.
- `line_item_value_total`: **190,908,939** calculated as `sum(Product Price × Quantity)`; not equal to recorded amount in **249,952** rows.
- `transactions`: **250,000** records, without a canonical order ID.
- `customers`: **49,673** distinct `Customer ID`.
- **Churn per customer**: `9,942 / 49,673 = 20.0149%`, instead of legacy **transaction-weighted 19.9496%**.
- **Observed Returns rate**: `100,769 / 202,404 = 49.7861%`; missing **47,596** records (19.0384%). Filling missing values with 0 would incorrectly display **40.3076%** on all rows.
- `Customer Name` is hidden in the Streamlit data table. Dashboard's charts and KPIs are based on the **same validated input file** and not the tracked legacy PNGs.

## Data ethics and interpretation

This dataset is synthetic; missingness mechanisms, currency, order semantics, and real-world generalization are **not established**. The original data shouldn't be silently corrected to make amount columns agree. Charts are descriptive, not causal or evidence of business intervention effects.

## Release gate

- Unit tests: **32/32 PASS** on fixture in the build environment (17 Phase 1–2 + 15 Phase 3).
- Full 250k chart functions smoke-tested on the original tracked values; **that is not a Phase 2 CSV acceptance run**.
- End-to-end dashboard and EDA export against the user's **actual** clean CSV are required before committing Phase 3. `docs/phase3_runbook.md` lists commands and expected outputs.
- Figures **06–09 remain unchanged** and will be reviewed in Phase 4 together with inference/regression.
