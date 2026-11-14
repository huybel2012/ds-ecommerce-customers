"""Rebuild W9 inference and W10 regression from Phase 2 clean CSV.

From repo root: python -m src.run_phase4 [--dry-run] [--only w9|w10]
Never uses the tracked legacy CSV or writes to data/raw or data/processed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from src.eda_metrics import CLEAN_PATH, load_cleaned
from src.utils import REPO_ROOT

MANIFEST=REPO_ROOT/'reports/phase4/run_manifest.json'
TASKS={'w9':'src.pipeline.07_hypothesis_tests','w10':'src.pipeline.06_regression'}


def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def main(args=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=('w9','w10'),help='Run one phase')
    parser.add_argument('--dry-run',action='store_true')
    opt=parser.parse_args(args)
    selected=[opt.only] if opt.only else list(TASKS)
    print(f'Phase 4: Python {sys.version.split()[0]}, source={CLEAN_PATH}',flush=True)
    for name in selected: print(f'  {name}: python -m {TASKS[name]}',flush=True)
    print('Source exists:',CLEAN_PATH.is_file(),flush=True)
    if opt.dry_run:return 0
    clean=load_cleaned()  # explicit schema+missingness validation; never use fallback legacy CSV
    count=len(clean)
    del clean
    original_sha=file_sha(CLEAN_PATH)
    for name in selected:
        subprocess.run([sys.executable,'-m',TASKS[name]],cwd=REPO_ROOT,check=True)
        if file_sha(CLEAN_PATH)!=original_sha:
            raise RuntimeError('Clean input was modified during Phase 4')
    targets={
      'w9':[REPO_ROOT/'reports/hypothesis_tests.md', REPO_ROOT/'reports/figures/09_hypothesis_tests.png',REPO_ROOT/'reports/phase4/hypothesis_results.json'],
      'w10':[REPO_ROOT/'reports/regression_results.md', REPO_ROOT/'reports/figures/06_residual_diagnostics.png',REPO_ROOT/'reports/figures/07_coefficients_ci.png',REPO_ROOT/'reports/figures/08_actual_vs_predicted.png',REPO_ROOT/'reports/phase4/regression_metrics.json']
    }
    artifacts={str(p.relative_to(REPO_ROOT)):file_sha(p) for name in selected for p in targets[name]}
    manifest={'phase':4,'completed_utc':datetime.now(timezone.utc).isoformat(),
              'python':sys.version.split()[0],'source':str(CLEAN_PATH.relative_to(REPO_ROOT)),
              'source_sha256':original_sha,'rows':count,'tasks':selected,'artifacts_sha256':artifacts,
              'limitations':'Synthetic data; currency and target semantics unverified.'}
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'PASS: Phase 4, source unchanged, {len(artifacts)} artifacts. Manifest: {MANIFEST}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
