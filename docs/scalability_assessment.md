# Phase 6 — Data Scale & Compute Assessment

- Current source: **250,000 synthetic transaction rows**, around **49,673 customers**; Phase 5 feature table: **49,128** customers × **19** features.
- Current pipeline uses Pandas/NumPy/scikit-learn and Parquet. This is appropriate for a laptop-scale course project without inventing Big Data requirements.
- Validate RAM footprint and wall-clock time on the team laptop before making performance claims; no benchmark to arbitrary 10-million-row data was run.
- Growth path: (1) retain immutable raw data and schema/data-quality gates; (2) use partitioned Parquet and pushdown instead of multiple full CSV copies; (3) aggregate customer features in chunks or query engine; (4) only evaluate distributed frameworks (Spark/Dask) if measured memory, latency or distributed storage requires it; (5) version input/output and model cards.
- The primary blockers are **label validity**, **synthetic generalizability**, **semantic inconsistency of monetary fields**, and **rights for redistributing data**, not the number of rows.
