"""W10 entrypoint. Run: python -m src.pipeline.06_regression."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression_phase4 import main

if __name__ == "__main__":
    main()
