"""Paths. Data never lives in the repository: it defaults to ./data (git-ignored) and can be moved with VOLDESK_DATA."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("VOLDESK_DATA", ROOT / "data")).resolve()
RAW = DATA / "raw"                 # vendor pulls exactly as received (parquet) + manifest.json
STORE = DATA / "store"             # cleaned, analysis-ready tables built from RAW
CACHE = DATA / "cache"             # intermediate results of long computations (fits, backtests)
RESULTS = ROOT / "results"         # small derived outputs that the reports cite (CSV/JSON, safe to publish)
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
WEB_BUNDLE = ROOT / "demo" / "data"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
